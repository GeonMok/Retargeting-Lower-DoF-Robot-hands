import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import smplx
import numpy as np
import cv2  
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
 
from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig
import trimesh.transformations as tf

from custom_optimizers import MyCustomPositionOptimizer

# =======================================================================
# Configuration & Paths
# =======================================================================
DATASET_DIR = "C:/4-1/kiat/Dataset/trajectory"
MODEL_PATH = "C:/4-1/KIAT/models"  
URDF_DIR = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf"
SDF_DIR = "./baked_sdfs"             # 구워진 SDF 파일들이 모여있는 폴더
OUTPUT_DIR = "./advanced_outputs"    # SDF가 적용된 궤적이 저장될 폴더
RETARGETING_CONFIG_PATH = "./custom_allegro_right_position.yml"

os.makedirs(OUTPUT_DIR, exist_ok=True)
RetargetingConfig.set_default_urdf_dir(URDF_DIR)

# =======================================================================
# Synergy Mapping (M_simple) & Constants
# =======================================================================
M_heuristic = torch.tensor([
        [0.836, 0.164, 0.000, 0.000, 0.000],
        [0.000, 0.720, 0.280, 0.000, 0.000],
        [0.000, 0.000, 0.260, 0.740, 0.000],
        [0.000, 0.000, 0.000, 0.160, 0.840]
    ], dtype=torch.float32)

tip_vertex_indices = [8079, 7669, 7794, 7905, 8022] 
R_offset = tf.rotation_matrix(np.pi/2, [-1, -1, 0])[:3, :3]

# =======================================================================
# Main Loop: Iterate over all Dataset files
# =======================================================================
npz_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
print(f"🔍 Found {len(npz_files)} dataset files to process with SDF Optimizer.\n")

for npz_path in npz_files:
    filename = os.path.basename(npz_path)
    obj_action = filename.replace(".npz", "") # e.g., 'apple_eat_1'
    obj_name = obj_action.split('_')[0]       # e.g., 'apple'
    
    print("==================================================")
    print(f" 📦 Processing: {obj_action} (Object: {obj_name})")
    print("==================================================")
    
    # 1. 동적 파일 경로 생성
    sdf_file_path = os.path.join(SDF_DIR, f"{obj_name}_sdf_res64.pt")
    output_filename = f"allegro_limitloss_s1_{obj_action}_nn.npy"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    
    # 해당 물체의 SDF 파일이 구워져 있지 않다면 스킵합니다.
    if not os.path.exists(sdf_file_path):
        print(f" ⚠️ Skipping {obj_action}: SDF file not found ({sdf_file_path})")
        continue

    # 2. 데이터 로드
    data = np.load(npz_path, allow_pickle=True)
    rhand_params = data['rhand'].item()['params']
    body_params = data['body'].item()['params']
    obj_params = data['object'].item()['params']
    
    obj_transl = obj_params['transl'] 
    obj_global_orient = obj_params['global_orient'] 
    n_frames = data['n_frames']
    gender = str(data['gender'])

    # 3. SMPL-X 초기화 (프레임 수가 다르므로 매번 새로 생성해야 함)
    hand_model = smplx.create(MODEL_PATH, model_type='smplx', gender=gender, 
                              use_pca=False, flat_hand_mean=True, batch_size=n_frames)

    transl = torch.tensor(body_params['transl'], dtype=torch.float32)
    global_orient = torch.tensor(body_params['global_orient'], dtype=torch.float32)
    body_pose = torch.tensor(body_params['body_pose'], dtype=torch.float32)
    right_hand_pose = torch.tensor(rhand_params['fullpose'], dtype=torch.float32)

    output = hand_model(transl=transl, global_orient=global_orient, 
                        body_pose=body_pose, right_hand_pose=right_hand_pose, return_verts=True)
    joints_3d = output.joints.detach().numpy()  
    verts_3d = output.vertices.detach().numpy() 

    # 4. Custom Optimizer 주입 (물체가 바뀌었으므로 SDF 엔진도 새로 교체!)
    config = get_retargeting_config(RETARGETING_CONFIG_PATH)
    retargeting = config.build() 
    default_optimizer = retargeting.optimizer

    my_custom_optimizer = MyCustomPositionOptimizer(
        sdf_path=sdf_file_path,            # 🚨 물체에 맞는 동적 SDF 경로 주입
        use_sdf=False,                      # SDF 스위치 
        penetration_threshold=0.005, 
        penetration_weight=2000.0,   
        use_attraction=False,
        contact_margin=0.01,
        attraction_weight=100.0,
        use_limit=True,                    # Limit 스위치 
        limit_margin=0.05,
        limit_weight=500.0,
        robot=default_optimizer.robot,
        target_joint_names=default_optimizer.target_joint_names,
        target_link_names=default_optimizer.body_names,
        target_link_human_indices=default_optimizer.target_link_human_indices,
        huber_delta=0.02,
        norm_delta=4e-3
    )
    retargeting.optimizer = my_custom_optimizer

    # 5. 프레임 최적화 루프
    trajectory_list = []
    for i in tqdm(range(n_frames), desc=f"   -> Optimizing [SDF + M_simple]"):
        current_obj_transl = obj_transl[i]
        current_obj_rot_matrix, _ = cv2.Rodrigues(obj_global_orient[i]) 
        
        my_custom_optimizer.update_object_pose(current_obj_transl, current_obj_rot_matrix)

        absolute_tips = verts_3d[i, tip_vertex_indices]
        human_wrist_pos = joints_3d[i, 21]
        M = joints_3d[i, 43]
        I = joints_3d[i, 40]

        y_axis = (M - human_wrist_pos) / np.linalg.norm(M - human_wrist_pos)
        temp_x = (I - M) / np.linalg.norm(I - M)
        z_axis = np.cross(temp_x, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        R_human = np.column_stack((x_axis, y_axis, z_axis))
        R_robot = R_human @ R_offset

        T_robot = np.eye(4); T_robot[:3, :3] = R_robot
        rx, ry, rz = tf.euler_from_matrix(T_robot, axes='sxyz')

        world_targets = absolute_tips - human_wrist_pos
        local_targets = world_targets @ R_robot
        P_human_targets = torch.tensor(local_targets, dtype=torch.float32)
        P_robot_targets = torch.matmul(M_heuristic, P_human_targets)
        numpy_targets = P_robot_targets.numpy().astype(np.float32)
        
        if i == 0:
            for _ in range(10): retargeting.retarget(numpy_targets)

        finger_qpos = retargeting.retarget(numpy_targets)
        
        robot_qpos_22 = np.zeros(22)
        robot_qpos_22[0:3] = human_wrist_pos    
        robot_qpos_22[3:6] = [rx, ry, rz]    
        robot_qpos_22[6:22] = finger_qpos
        trajectory_list.append(robot_qpos_22)

    trajectory_array = np.stack(trajectory_list, axis=0)
    np.save(output_filepath, trajectory_array)
    print(f" ✅ Saved: {output_filename}\n")

print("🎉 ALL DATASETS PROCESSED SUCCESSFULLY! 🎉")