# Baseline Trajectory Generation Script under existing loss functions (No SDF or Limit Loss)
# Experiment on the Mapping Matrix

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import smplx
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
 
from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig
import trimesh.transformations as tf

USE_VECTOR_MODE = False  # True: Vector, False: Position 
USE_DROP_MAPPING = False   # 🚨 True: M_drop, False: M_simple

if USE_VECTOR_MODE:
    RETARGETING_CONFIG_PATH = "./custom_allegro_right_vector.yml"
    print("🚀 MODE: VECTOR RETARGETING")
else:
    RETARGETING_CONFIG_PATH = "./custom_allegro_right_position.yml"
    print("🎯 MODE: POSITION RETARGETING")

# =======================================================================
# Configuration & Paths
# =======================================================================
MODEL_PATH = "C:/4-1/KIAT/models"  
NPZ_FILE = 'C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/apple_eat_1.npz'
URDF_DIR = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf"

mapping_str = "drop" if USE_DROP_MAPPING else "simple"
OUTPUT_FILE = f"allegro_trajectory_s1_apple_eat_1_{mapping_str}.npy"

# Set default URDF directory for dex-retargeting
RetargetingConfig.set_default_urdf_dir(URDF_DIR)

# =======================================================================
# Phase 1: Engine Initialization & Human Data Extraction
# =======================================================================
print("=== 1. Initializing Engines & Extracting Data ===")

# 1-A. Load Retargeting Optimizer
config = get_retargeting_config(RETARGETING_CONFIG_PATH)
retargeting = config.build()
print("-> Dex-Retargeting Optimizer loaded.")

# 1-B. Load GRAB Data
data = np.load(NPZ_FILE, allow_pickle=True)
rhand_params = data['rhand'].item()['params']
body_params = data['body'].item()['params']
n_frames = data['n_frames']
gender = str(data['gender'])

# 1-C. Initialize SMPL-X
hand_model = smplx.create(MODEL_PATH, model_type='smplx', gender=gender, 
                          use_pca=False, flat_hand_mean=True, batch_size=n_frames)

transl = torch.tensor(body_params['transl'], dtype=torch.float32)
global_orient = torch.tensor(body_params['global_orient'], dtype=torch.float32)
body_pose = torch.tensor(body_params['body_pose'], dtype=torch.float32)
right_hand_pose = torch.tensor(rhand_params['fullpose'], dtype=torch.float32)

output = hand_model(
    transl=transl, 
    global_orient=global_orient, 
    body_pose=body_pose, 
    right_hand_pose=right_hand_pose, 
    return_verts=True
)
joints_3d = output.joints.detach().numpy()  # (N, 127, 3)
verts_3d = output.vertices.detach().numpy() # (N, 10475, 3)

print(f"-> SMPL-X Processing Complete. Total Frames: {n_frames}")


# =======================================================================
# Phase 2: Topology Mapping Matrix Setup
# =======================================================================
# Heuristic Synergy Matrix M (4x5)
if USE_DROP_MAPPING:
    M_heuristic = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    print(f"\n-> 🧮 Using M_drop mapping (Pinky dropped)")
else:
    M_heuristic = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5]
    ], dtype=torch.float32)
    print(f"\n-> 🧮 Using M_simple mapping (Ring & Pinky synergy)")

tip_vertex_indices = [8079, 7669, 7794, 7905, 8022] # rthumb, rindex, rmiddle, rring, rpinky 
R_offset = tf.rotation_matrix(np.pi/2, [-1, -1, 0])[:3, :3]

# =======================================================================
# Phase 3: Optimization
# =======================================================================
print("\n=== 3. Optimization Loop ===")
trajectory_list = []

for i in tqdm(range(n_frames), desc="Optimizing Trajectory"):
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
    
    # 🔥 Warm-up for Frame 0
    if i == 0:
        for _ in range(10): retargeting.retarget(numpy_targets)

    finger_qpos = retargeting.retarget(numpy_targets)
    
    robot_qpos_22 = np.zeros(22)
    robot_qpos_22[0:3] = human_wrist_pos    
    robot_qpos_22[3:6] = [rx, ry, rz]    
    robot_qpos_22[6:22] = finger_qpos
    trajectory_list.append(robot_qpos_22)

trajectory_array = np.stack(trajectory_list, axis=0)
np.save(OUTPUT_FILE, trajectory_array)
print(f"\n🎉 SUCCESS! Saved to '{OUTPUT_FILE}'")