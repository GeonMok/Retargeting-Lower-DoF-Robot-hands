import numpy as np
import trimesh
import trimesh.transformations as tf
from yourdfpy import URDF
import os
import time
import cv2
import warnings
import smplx
import torch
import sapien.core as sapien # Import Sapien Engine
warnings.filterwarnings('ignore')

print("=== 1. Setting Up Absolute Paths ===")
SMPLX_MODEL_PATH = "C:/4-1/KIAT/models"
TRAJECTORY_FILE = "C:/4-1/KIAT/Preprocessing/allegro_trajectory_s1_apple_eat_1.npy" 
APPLE_MESH_FILE = "C:/4-1/KIAT/GRAB/dataset_unzipped/tools/object_meshes/contact_meshes/apple.ply"
GRAB_NPZ_FILE = "C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/apple_eat_1.npz"
URDF_PATH = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf/allegro_hand_description_right.urdf"

ALLEGRO_JOINT_NAMES = [
    'joint_0', 'joint_1', 'joint_2', 'joint_3',       
    'joint_4', 'joint_5', 'joint_6', 'joint_7',       
    'joint_8', 'joint_9', 'joint_10', 'joint_11',     
    'joint_12', 'joint_13', 'joint_14', 'joint_15'    
]

print("\n=== 2. Loading Data ===")
trajectory = np.load(TRAJECTORY_FILE)
n_frames = trajectory.shape[0]

robot_urdf = URDF.load(URDF_PATH)
apple_mesh = trimesh.load(APPLE_MESH_FILE)
apple_mesh.visual.face_colors = [100, 255, 100, 150] 

grab_data = np.load(GRAB_NPZ_FILE, allow_pickle=True)
obj_data = grab_data['object'].item()['params']
obj_transl = obj_data['transl']          
obj_global_orient = obj_data['global_orient'] 

sync_frames = min(n_frames, obj_transl.shape[0])
print(f"-> Synced Frames: {sync_frames}")

# 🚨 2-C. Reconstruct Human SMPL-X Mesh (Ground Truth)
print("-> Reconstructing Human Hand Mesh from SMPL-X params...")
body_data = grab_data['body'].item()['params']
gender = str(grab_data['gender'])

# Initialize SMPL-X model (calculator)
# 🚨 Must match parameters used in generate_trajectory.py!
human_model = smplx.create(
    SMPLX_MODEL_PATH, model_type='smplx', gender=gender,
    use_pca=True, num_pca_comps=24, flat_hand_mean=True, batch_size=sync_frames
)

# Move parameters to torch tensors
transl_t = torch.tensor(body_data['transl'][:sync_frames], dtype=torch.float32)
global_orient_t = torch.tensor(body_data['global_orient'][:sync_frames], dtype=torch.float32)
body_pose_t = torch.tensor(body_data['body_pose'][:sync_frames], dtype=torch.float32)
right_hand_pose_t = torch.tensor(body_data['right_hand_pose'][:sync_frames], dtype=torch.float32)

# Run forward pass to get vertices
output = human_model(
    transl=transl_t, 
    global_orient=global_orient_t, 
    body_pose=body_pose_t, 
    right_hand_pose=right_hand_pose_t, 
    return_verts=True
)
human_verts_3d = output.vertices.detach().numpy() # (N, 10475, 3)

# 🚨 속도 최적화 수술: 오른쪽 손목 주변만 잘라냅니다 (Mesh Cropping)
wrist_pos_0 = output.joints[0, 21].detach().numpy() # 프레임 0의 오른쪽 손목 위치
# 손목에서 25cm 이내에 있는 점들만 True로 마스킹
hand_vertex_mask = np.linalg.norm(human_verts_3d[0] - wrist_pos_0, axis=1) < 0.25 
# 손 점이 하나라도 포함된 삼각형(Face)들만 추출
hand_faces_mask = hand_vertex_mask[human_model.faces].any(axis=1)
hand_faces_only = human_model.faces[hand_faces_mask]

# 가벼워진 전용 메쉬 생성!
human_mesh_template = trimesh.Trimesh(
    vertices=human_verts_3d[0], 
    faces=hand_faces_only,  # 🚨 전신이 아닌 잘라낸 손 Faces만 입력
    process=False
)
# Set human hand color to semi-transparent blue/grey for contrast
human_mesh_template.visual.face_colors = [150, 150, 200, 100] # Very transparent

print("-> Human Hand Reconstruction Complete.")


print("\n=== 3. Starting Scene Construction ===")
combined_scene = trimesh.Scene()
combined_scene.add_geometry(apple_mesh, node_name='apple_node')
# 🚨 Add the dynamic human hand mesh node
combined_scene.add_geometry(human_mesh_template, geom_name='human_hand_geom', node_name='human_hand_node')

for node_name in robot_urdf.scene.graph.nodes_geometry:
    local_transform, geom_name = robot_urdf.scene.graph[node_name]
    geom = robot_urdf.scene.geometry[geom_name]
    geom.visual.face_colors = [255, 100, 100, 180] 
    combined_scene.add_geometry(geom, node_name=f"robot_{node_name}", transform=local_transform)

print("\n=== 4. Setting Initial Camera View (Frame 0) ===")
# 🚨 해결책: 뷰어가 켜지기 "전에" 프레임 0의 위치로 모든 물체를 세팅합니다!
initial_qpos = trajectory[0]
wrist_transl_xyz_0 = initial_qpos[0:3]     
wrist_rot_rpy_0 = initial_qpos[3:6]       
finger_angles_0 = initial_qpos[6:22]      

initial_obj_transl_0 = obj_transl[0]
initial_obj_orient_aa_0 = obj_global_orient[0] 

# 🚨 행렬 변환 (R_viz_align 제거!)
cv2.Rodrigues(initial_obj_orient_aa_0)[0]
    
T_wrist_0 = np.dot(tf.translation_matrix(wrist_transl_xyz_0), 
                   tf.euler_matrix(wrist_rot_rpy_0[0], wrist_rot_rpy_0[1], wrist_rot_rpy_0[2], axes='sxyz'))

for node_name in robot_urdf.scene.graph.nodes_geometry:
    local_transform, _ = robot_urdf.scene.graph[node_name]
    combined_scene.graph.update(f"robot_{node_name}", matrix=np.dot(T_wrist_0, local_transform))
    
# 사과 위치도 프레임 0으로 업데이트
combined_scene.graph.update("apple_node", matrix=T_wrist_0) # 🚨 임시로 손목 좌표를 주어 뷰어 포커싱을 유도합니다!

# 카메라 포커스를 사과와 손목 중앙으로! (흰 화면 완벽 해결)
combined_scene.set_camera(distance=1.0, center=initial_obj_transl_0)


print("\n=== 5. Launching Animated Viewer ===")
frame_idx = 0


def update_callback(scene):
    global frame_idx, sync_frames, apple_mesh
    time.sleep(1/30.0)
    
    qpos = trajectory[frame_idx]
    wrist_transl_xyz = qpos[0:3]     
    wrist_rot_rpy = qpos[3:6]       
    finger_angles = qpos[6:22]      
    
    current_obj_transl = obj_transl[frame_idx]
    current_obj_orient_aa = obj_global_orient[frame_idx] 

    # 2. 행렬 변환
    current_obj_rot_matrix, _ = cv2.Rodrigues(current_obj_orient_aa)
    
    T_wrist_transl = tf.translation_matrix(wrist_transl_xyz)
    T_wrist_rot = tf.euler_matrix(wrist_rot_rpy[0], wrist_rot_rpy[1], wrist_rot_rpy[2], axes='sxyz')
    T_wrist = np.dot(T_wrist_transl, T_wrist_rot)
    
    T_object_transl = tf.translation_matrix(current_obj_transl)
    T_object_rot = np.eye(4)
    T_object_rot[:3, :3] = current_obj_rot_matrix
    T_object = np.dot(T_object_transl, T_object_rot)

    # 3. 역기구학 업데이트
    joint_dict = {name: angle for name, angle in zip(ALLEGRO_JOINT_NAMES, finger_angles)}
    robot_urdf.update_cfg(joint_dict) 

    # --- 🚀 3. Human (Ground Truth) Mesh Update ---
    # Update the vertices of the human mesh geometry *in place* for performance.
    # We do not need transforms because SMPL-X output is already in Absolute World coords.
    scene.geometry['human_hand_geom'].vertices = human_verts_3d[frame_idx]

    for node_name in robot_urdf.scene.graph.nodes_geometry:
        local_transform, _ = robot_urdf.scene.graph[node_name]
        final_transform = np.dot(T_wrist, local_transform)
        scene.graph.update(f"robot_{node_name}", matrix=final_transform)
        
    scene.graph.update("apple_node", matrix=T_object)

    frame_idx = (frame_idx + 1) % sync_frames

# 창 크기 기본 설정
combined_scene.show(callback=update_callback, resolution=(1024, 768))