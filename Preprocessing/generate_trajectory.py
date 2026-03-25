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

# =======================================================================
# Configuration & Paths
# =======================================================================
MODEL_PATH = "C:/4-1/KIAT/models"  
NPZ_FILE = 'C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/apple_eat_1.npz'
RETARGETING_CONFIG_PATH = "C:/4-1/KIAT/dex-retargeting/src/dex_retargeting/configs/offline/custom_allegro_hand_right.yml"
URDF_DIR = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf"
OUTPUT_FILE = "allegro_trajectory_s1_apple_eat_1.npy"

# Set default URDF directory for dex-retargeting
RetargetingConfig.set_default_urdf_dir(URDF_DIR)

# True: World Coordinate System (Visualization, SDF calculation, etc.) 
# False: Robot Base Coordinate System (Real Robot Control, Grasp analysis, etc.)
USE_ABSOLUTE_COORD = True

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
#rhand_params = data['rhand'].item()['params']
body_params = data['body'].item()['params']
n_frames = data['n_frames']
gender = str(data['gender'])

# 1-C. Initialize SMPL-X
hand_model = smplx.create(MODEL_PATH, model_type='smplx', gender=gender, 
                          use_pca=True, num_pca_comps=24, flat_hand_mean=True, batch_size=n_frames)

# fullpose = torch.tensor(rhand_params['fullpose'], dtype=torch.float32)
# global_orient = torch.tensor(rhand_params['global_orient'], dtype=torch.float32)
# transl = torch.tensor(rhand_params['transl'], dtype=torch.float32)
transl = torch.tensor(body_params['transl'], dtype=torch.float32)
global_orient = torch.tensor(body_params['global_orient'], dtype=torch.float32)
body_pose = torch.tensor(body_params['body_pose'], dtype=torch.float32)
right_hand_pose = torch.tensor(body_params['right_hand_pose'], dtype=torch.float32)

# # Forward Kinematics for ALL frames simultaneously
# output = hand_model(global_orient=global_orient, right_hand_pose=fullpose, transl=transl, return_verts=True)
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
M_heuristic = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0],  # Thumb
    [0.0, 1.0, 0.0, 0.0, 0.0],  # Index
    [0.0, 0.0, 1.0, 0.0, 0.0],  # Middle
    [0.0, 0.0, 0.0, 0.5, 0.5],  # Ring (50%) + Pinky (50%)
], dtype=torch.float32)


tip_vertex_indices = [8079, 7669, 7794, 7905, 8022] # rthumb, rindex, rmiddle, rring, rpinky 
middle_joint_indices = [53, 41, 44, 50, 47] # right_thumb2, right_index2, right_middle2, right_ring2, right_pinky2
base_joint_indices = [52, 40, 43, 49, 46] # right_thumb1, right_index1, right_middle1, right_ring1, right_pinky1

# =======================================================================
# Phase 3: Warm-up (Eliminating Initialization Jump)
# =======================================================================
print("\n=== 3. Warming up the Optimizer (Pre-computing Frame 0) ===")
# [vector]
absolute_tips_0 = verts_3d[0, tip_vertex_indices] 
absolute_bases_0 = joints_3d[0, base_joint_indices]

human_vectors_0 = absolute_tips_0 - absolute_bases_0
P_human_vectors_0 = torch.tensor(human_vectors_0, dtype=torch.float32)

P_robot_vectors_0 = torch.matmul(M_heuristic, P_human_vectors_0)
target_vectors_0 = P_robot_vectors_0.numpy().astype(np.float32)

for _ in range(10):
    retargeting.retarget(target_vectors_0)

# [position]
# # Extract targets specifically for Frame 0
# wrist_pos_0 = joints_3d[0, 21]
# absolute_tips_0 = verts_3d[0, tip_vertex_indices] 
# absolute_middles_0 = joints_3d[0, middle_joint_indices]

# if USE_ABSOLUTE_COORD:
#     human_tips_0 = absolute_tips_0
#     human_middles_0 = absolute_middles_0
# else:
#     human_tips_0 = absolute_tips_0 - wrist_pos_0
#     human_middles_0 = absolute_middles_0 - wrist_pos_0

# P_human_tips_0 = torch.tensor(human_tips_0, dtype=torch.float32)
# P_human_middles_0 = torch.tensor(human_middles_0, dtype=torch.float32)

# P_robot_tips_0 = torch.matmul(M_heuristic, P_human_tips_0)
# P_robot_middles_0 = torch.matmul(M_heuristic, P_human_middles_0)

# # P_robot_8_targets_0 = torch.cat([P_robot_tips_0, P_robot_middles_0], dim=0)
# # target_positions_0 = P_robot_8_targets_0.numpy().astype(np.float32)

# target_positions_0 = P_robot_tips_0.numpy().astype(np.float32) # Only using fingertips

# # Run the optimizer 10 times off-camera to let the joints settle smoothly
# for _ in range(10):
#     retargeting.retarget(target_positions_0)

# =======================================================================
# Phase 4: Frame-by-Frame Optimization Loop
# =======================================================================
print("\n=== 4. Starting Frame-by-Frame Optimization ===")

# Container to store the calculated joint angles for all frames
trajectory_list = []

# [vector]
for i in tqdm(range(n_frames), desc="Optimizing Trajectory"):
    absolute_tips = verts_3d[i, tip_vertex_indices]
    absolute_bases = joints_3d[i, base_joint_indices]
    
    human_vectors = absolute_tips - absolute_bases
    P_human_vectors = torch.tensor(human_vectors, dtype=torch.float32)
    
    P_robot_vectors = torch.matmul(M_heuristic, P_human_vectors)
    target_vectors = P_robot_vectors.numpy().astype(np.float32)
    
    robot_qpos = retargeting.retarget(target_vectors)

    human_wrist_pos = joints_3d[i, 21]
    robot_qpos[0:3] = human_wrist_pos  # Set the first 3 DoF to match the wrist position (absolute coordinates)
    trajectory_list.append(robot_qpos)

# [position]
# # Loop through each frame with a progress bar
# for i in tqdm(range(n_frames), desc="Optimizing Trajectory"):
    
#     # Extract relative positions for the current frame
#     wrist_pos = joints_3d[i, 21]
#     absolute_tips = verts_3d[i, tip_vertex_indices]
#     absolute_middles = joints_3d[i, middle_joint_indices]

#     if USE_ABSOLUTE_COORD:
#         human_tips = absolute_tips
#         human_middles = absolute_middles
#     else:
#         human_tips = absolute_tips - wrist_pos
#         human_middles = absolute_middles - wrist_pos

#     P_human_tips = torch.tensor(human_tips, dtype=torch.float32)
#     P_human_middles = torch.tensor(human_middles, dtype=torch.float32)
    
#     P_robot_tips = torch.matmul(M_heuristic, P_human_tips)
#     P_robot_middles = torch.matmul(M_heuristic, P_human_middles)
    
#     # P_robot_8_targets = torch.cat([P_robot_tips, P_robot_middles], dim=0)
#     # target_positions = P_robot_8_targets.numpy().astype(np.float32)

#     target_positions = P_robot_tips.numpy().astype(np.float32) # Only using fingertips

    
#     robot_qpos = retargeting.retarget(target_positions)
#     trajectory_list.append(robot_qpos)

# Convert list to a numpy array: shape should be (1628, 22)
trajectory_array = np.stack(trajectory_list, axis=0)

# =======================================================================
# Phase 4: Save Dataset
# =======================================================================
np.save(OUTPUT_FILE, trajectory_array)
print(f"\n🎉 SUCCESS! Trajectory saved to '{OUTPUT_FILE}' 🎉")
print(f"Final Dataset Shape: {trajectory_array.shape} (Frames, DoF)")