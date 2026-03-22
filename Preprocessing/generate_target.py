# This script is 'one frame' version of 'generate_trajectory.py' including visualizatoin for debugging the entire pipeline step by step.

import os
# Suppress OpenMP duplicate library loading error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import smplx
import numpy as n
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig


# Paths
MODEL_PATH = "C:/4-1/KIAT/models"  
NPZ_FILE = 'C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/apple_eat_1.npz'
RETARGETING_CONFIG_PATH = "C:/4-1/KIAT/dex-retargeting/src/dex_retargeting/configs/offline/custom_allegro_hand_right.yml"

def get_human_keypoints_from_grab(npz_path, model_path):
    # 1. Load GRAB data
    data = np.load(npz_path, allow_pickle=True)
    rhand_params = data['rhand'].item()['params']
    
    n_frames = data['n_frames']
    gender = str(data['gender'])
    # 2. Load SMPL-X model and generate 3D joints & vertices
    hand_model = smplx.create(model_path, 
                              model_type='smplx',
                              gender=gender, 
                              use_pca=False,
                              flat_hand_mean=True,
                              batch_size=n_frames)

    # Convert GRAB parameters to PyTorch tensors
    fullpose = torch.tensor(rhand_params['fullpose'], dtype=torch.float32)
    global_orient = torch.tensor(rhand_params['global_orient'], dtype=torch.float32)
    transl = torch.tensor(rhand_params['transl'], dtype=torch.float32)

    # Perform Forward Kinematics to get 3D joints and vertices
    output = hand_model(
        global_orient=global_orient,
        right_hand_pose=fullpose,
        transl=transl,
        return_verts=True # not only joints but also vertices
    )

    joints_3d = output.joints.detach().numpy() # (N, 127, 3): frames, joints, XYZ
    verts_3d = output.vertices.detach().numpy() # (N, 10475, 3): frames, vertices, XYZ

    return joints_3d, verts_3d, data

# =======================================================================
# Phase 1: Human Data Extraction
# =======================================================================


print("=== 1. Human Data Extraction ===")
joints, vertices, raw_data = get_human_keypoints_from_grab(NPZ_FILE, MODEL_PATH)
print(f"Joints shape: {joints.shape}, Vertices shape: {vertices.shape}") 

# Get the wrist position for the first frame (Index 21 in SMPL-X)
wrist_pos = joints[0, 21]

# We use vertex positions of the fingertips instead of joint positions for better accuracy (https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py).
tip_vertex_indices = [8079, 7669, 7794, 7905, 8022] 
relative_tips = vertices[0, tip_vertex_indices] - wrist_pos

middle_joint_indices = [53, 41, 44, 50, 47]
relative_middles = joints[0, middle_joint_indices] - wrist_pos

# =======================================================================
# Phase 2: Topology Mapping (Cross-Embodiment)
# =======================================================================
print("\n=== 2. Cross-Embodiment Topology Mapping ===")

# Mapping matrix M (4 x 5): Maps 5 human fingers to 4 robot fingers
M_heuristic = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0],  # Robot Thumb  <- Human Thumb
    [0.0, 1.0, 0.0, 0.0, 0.0],  # Robot Index  <- Human Index
    [0.0, 0.0, 1.0, 0.0, 0.0],  # Robot Middle <- Human Middle
    [0.0, 0.0, 0.0, 0.5, 0.5],  # Robot Ring   <- Human Ring (50%) + Pinky (50%)
], dtype=torch.float32)

P_human_tips = torch.tensor(relative_tips, dtype=torch.float32)       # (5, 3)
P_human_middles = torch.tensor(relative_middles, dtype=torch.float32) # (5, 3)

# Matrix Multiplication: (4 x 5) @ (5 x 3) = (4 x 3)
P_robot_tips = torch.matmul(M_heuristic, P_human_tips)       # (4, 3)
P_robot_middles = torch.matmul(M_heuristic, P_human_middles) # (4, 3)

P_robot_8_targets = torch.cat([P_robot_tips, P_robot_middles], dim=0) # (8, 3)
print(f"P_robot (8x3 targets):\n{P_robot_8_targets.numpy()}\n")

# =======================================================================
# Phase 3: Inverse Kinematics (Dex-Retargeting Optimizer)
# =======================================================================
print("=== 3. Inverse Kinematics (Robot Joint Angle Extraction) ===")

URDF_DIR = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf"
RetargetingConfig.set_default_urdf_dir(URDF_DIR)

# 1. Load the optimizer configuration
config = get_retargeting_config(RETARGETING_CONFIG_PATH)
retargeting = config.build()
print("Optimizer engine loaded successfully.")

# 2. Prepare the inputs for the optimizer
target_positions_for_optimizer = P_robot_8_targets.numpy().astype(np.float32)

# 3. Directly call the retargeting function to get joint angles
print("Running optimization to find 16 joint angles...")
robot_qpos = retargeting.retarget(target_positions_for_optimizer)

# 4. Display the results
print("\n🎉 SUCCESS! Extracted Allegro Hand 16 Joint Angles (Radians) 🎉")
np.set_printoptions(precision=4, suppress=True)
print(robot_qpos) # 16 + 6 (free joints) = 22 total DOF

# =======================================================================
# Phase 4: Visualization for Debugging
# =======================================================================

print("\n=== 4. Visual Debugging (3D Plot of Topology Mapping) ===")

# Convert PyTorch tensors to Numpy arrays for plotting
h_tips = P_human_tips.numpy()       # Human 5 tips (5, 3)
h_mids = P_human_middles.numpy()    # Human 5 middle joints (5, 3)
r_tips = P_robot_tips.numpy()       # Robot 4 tips (4, 3)
r_mids = P_robot_middles.numpy()    # Robot 4 middle joints (4, 3)
wrist = np.array([0, 0, 0])         # Wrist is at origin (0, 0, 0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot Human Hand (Blue)
ax.scatter(h_tips[:, 0], h_tips[:, 1], h_tips[:, 2], c='blue', marker='o', s=50, label='Human Tips (5)')
ax.scatter(h_mids[:, 0], h_mids[:, 1], h_mids[:, 2], c='cyan', marker='o', s=30, label='Human Middles (5)')

# Draw lines for Human fingers (Wrist -> Middle -> Tip)
for i in range(5):
    ax.plot([wrist[0], h_mids[i, 0], h_tips[i, 0]], 
            [wrist[1], h_mids[i, 1], h_tips[i, 1]], color='blue', alpha=0.3)

# 2. Plot Robot Targets (Red)
# To show them clearly, we add a very small offset to the robot points so they don't completely overlap with human points
offset = 0.002 
ax.scatter(r_tips[:, 0]+offset, r_tips[:, 1]+offset, r_tips[:, 2]+offset, c='red', marker='^', s=80, label='Robot Targets (4)')
ax.scatter(r_mids[:, 0]+offset, r_mids[:, 1]+offset, r_mids[:, 2]+offset, c='orange', marker='^', s=50, label='Robot Middle Targets (4)')

# Draw lines for Robot fingers (Wrist -> Middle -> Tip)
for i in range(4):
    ax.plot([wrist[0], r_mids[i, 0]+offset, r_tips[i, 0]+offset], 
            [wrist[1], r_mids[i, 1]+offset, r_tips[i, 1]+offset], color='red', alpha=0.6, linestyle='--')

# 3. Set Plot Properties
ax.set_xlabel('X (Meters)')
ax.set_ylabel('Y (Meters)')
ax.set_zlabel('Z (Meters)')
ax.set_title('Cross-Embodiment Topology Mapping Verification')

# Make axes scale equal for realistic proportions
max_range = np.array([h_tips[:,0].max()-h_tips[:,0].min(), h_tips[:,1].max()-h_tips[:,1].min(), h_tips[:,2].max()-h_tips[:,2].min()]).max() / 2.0
mid_x = (h_tips[:,0].max()+h_tips[:,0].min()) * 0.5
mid_y = (h_tips[:,1].max()+h_tips[:,1].min()) * 0.5
mid_z = (h_tips[:,2].max()+h_tips[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.legend()
plt.show()