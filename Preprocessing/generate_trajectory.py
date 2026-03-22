import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import smplx
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig

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
n_frames = data['n_frames']
gender = str(data['gender'])

# 1-C. Initialize SMPL-X
hand_model = smplx.create(MODEL_PATH, model_type='smplx', gender=gender, 
                          use_pca=False, flat_hand_mean=True, batch_size=n_frames)

fullpose = torch.tensor(rhand_params['fullpose'], dtype=torch.float32)
global_orient = torch.tensor(rhand_params['global_orient'], dtype=torch.float32)
transl = torch.tensor(rhand_params['transl'], dtype=torch.float32)

# Forward Kinematics for ALL frames simultaneously
output = hand_model(global_orient=global_orient, right_hand_pose=fullpose, transl=transl, return_verts=True)
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

tip_vertex_indices = [8079, 7669, 7794, 7905, 8022] 
middle_joint_indices = [53, 41, 44, 50, 47]

# =======================================================================
# Phase 3: Warm-up (Eliminating Initialization Jump)
# =======================================================================
print("\n=== 3. Warming up the Optimizer (Pre-computing Frame 0) ===")

# Extract targets specifically for Frame 0
wrist_pos_0 = joints_3d[0, 21]
relative_tips_0 = verts_3d[0, tip_vertex_indices] - wrist_pos_0
relative_middles_0 = joints_3d[0, middle_joint_indices] - wrist_pos_0

P_human_tips_0 = torch.tensor(relative_tips_0, dtype=torch.float32)
P_human_middles_0 = torch.tensor(relative_middles_0, dtype=torch.float32)

P_robot_tips_0 = torch.matmul(M_heuristic, P_human_tips_0)
P_robot_middles_0 = torch.matmul(M_heuristic, P_human_middles_0)

P_robot_8_targets_0 = torch.cat([P_robot_tips_0, P_robot_middles_0], dim=0)
target_positions_0 = P_robot_8_targets_0.numpy().astype(np.float32)

# Run the optimizer 10 times off-camera to let the joints settle smoothly
for _ in range(10):
    retargeting.retarget(target_positions_0)
    
print("-> Warm-up complete! The robot is now in the perfect starting pose.")

# =======================================================================
# Phase 4: Frame-by-Frame Optimization Loop
# =======================================================================
print("\n=== 4. Starting Frame-by-Frame Optimization ===")

# Container to store the calculated joint angles for all frames
trajectory_list = []

# Loop through each frame with a progress bar
for i in tqdm(range(n_frames), desc="Optimizing Trajectory"):
    
    # Extract relative positions for the current frame
    wrist_pos = joints_3d[i, 21]
    relative_tips = verts_3d[i, tip_vertex_indices] - wrist_pos
    relative_middles = joints_3d[i, middle_joint_indices] - wrist_pos
    
    # Convert to Tensors
    P_human_tips = torch.tensor(relative_tips, dtype=torch.float32)
    P_human_middles = torch.tensor(relative_middles, dtype=torch.float32)
    
    # Apply Cross-Embodiment Mapping
    P_robot_tips = torch.matmul(M_heuristic, P_human_tips)
    P_robot_middles = torch.matmul(M_heuristic, P_human_middles)
    
    # Combine tips and middles (8 Targets total)
    P_robot_8_targets = torch.cat([P_robot_tips, P_robot_middles], dim=0)
    target_positions = P_robot_8_targets.numpy().astype(np.float32)
    
    # Run Inverse Kinematics
    # as the starting point (last_qpos) for the next frame, ensuring smooth motion!
    robot_qpos = retargeting.retarget(target_positions)
    
    # Save the 22 DoF result
    trajectory_list.append(robot_qpos)

# Convert list to a numpy array: shape should be (1628, 22)
trajectory_array = np.stack(trajectory_list, axis=0)

# =======================================================================
# Phase 4: Save Dataset
# =======================================================================
np.save(OUTPUT_FILE, trajectory_array)
print(f"\n🎉 SUCCESS! Trajectory saved to '{OUTPUT_FILE}' 🎉")
print(f"Final Dataset Shape: {trajectory_array.shape} (Frames, DoF)")