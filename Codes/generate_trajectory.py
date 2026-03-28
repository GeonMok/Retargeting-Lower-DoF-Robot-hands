import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import smplx
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
 
from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig
from dex_retargeting.optimizer import PositionOptimizer, VectorOptimizer
import trimesh.transformations as tf

USE_VECTOR_MODE = False  # True: Vector, False: Position 

if USE_VECTOR_MODE:
    RETARGETING_CONFIG_PATH = "./custom_allegro_right_vector.yml"
    print("🚀 MODE: VECTOR RETARGETING")
else:
    RETARGETING_CONFIG_PATH = "./custom_allegro_right_position.yml"
    print("🎯 MODE: POSITION RETARGETING")

class MyCustomPositionOptimizer(PositionOptimizer):
    def get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        
        return super().get_objective_function(target_pos, fixed_qpos, last_qpos)

class MyCustomVectorOptimizer(VectorOptimizer):
    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        return super().get_objective_function(target_vector, fixed_qpos, last_qpos)

# =======================================================================
# Configuration & Paths
# =======================================================================
MODEL_PATH = "C:/4-1/KIAT/models"  
NPZ_FILE = 'C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/apple_eat_1.npz'
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

default_optimizer = retargeting.optimizer

if USE_VECTOR_MODE:
    my_custom_optimizer = MyCustomVectorOptimizer(
        robot=default_optimizer.robot,
        target_joint_names=default_optimizer.target_joint_names,
        target_origin_link_names=default_optimizer.origin_link_names,
        target_task_link_names=default_optimizer.task_link_names,
        target_link_human_indices=default_optimizer.target_link_human_indices,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=default_optimizer.scaling # only for vector mode
    )
    print("-> 🛠️ Custom VECTOR Optimizer injected!")
else:
    my_custom_optimizer = MyCustomPositionOptimizer(
        robot=default_optimizer.robot,
        target_joint_names=default_optimizer.target_joint_names,
        target_link_names=default_optimizer.body_names,
        target_link_human_indices=default_optimizer.target_link_human_indices,
        huber_delta=0.02,
        norm_delta=4e-3
    )
    print("-> 🛠️ Custom POSITION Optimizer injected!")

retargeting.optimizer = my_custom_optimizer

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
M_heuristic = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0],  # Thumb
    [0.0, 1.0, 0.0, 0.0, 0.0],  # Index
    [0.0, 0.0, 1.0, 0.0, 0.0],  # Middle
    [0.0, 0.0, 0.0, 0.5, 0.5],  # Ring (50%) + Pinky (50%)
], dtype=torch.float32)


tip_vertex_indices = [8079, 7669, 7794, 7905, 8022] # rthumb, rindex, rmiddle, rring, rpinky 
middle_joint_indices = [53, 41, 44, 50, 47] # right_thumb2, right_index2, right_middle2, right_ring2, right_pinky2
base_joint_indices = [52, 40, 43, 49, 46] # right_thumb1, right_index1, right_middle1, right_ring1, right_pinky1
R_offset = tf.rotation_matrix(np.pi/2, [-1, -1, 0])[:3, :3]

# =======================================================================
# Phase 3: Warm-up (Eliminating Initialization Jump)
# =======================================================================
print("\n=== 3. Warming up the Optimizer (Pre-computing Frame 0) ===")
# [vector]
absolute_tips_0 = verts_3d[0, tip_vertex_indices] 
wrist_pos_0 = joints_3d[0, 21] 

M_0 = joints_3d[0, 43] # middle1
I_0 = joints_3d[0, 40] # index1

# Rotation Matrix R_human_0 (Hand Coordinate System)
y_axis_0 = (M_0 - wrist_pos_0) / np.linalg.norm(M_0 - wrist_pos_0)
temp_x_0 = (I_0 - M_0) / np.linalg.norm(I_0 - M_0)
z_axis_0 = np.cross(temp_x_0, y_axis_0)
z_axis_0 /= np.linalg.norm(z_axis_0)
x_axis_0 = np.cross(y_axis_0, z_axis_0)

R_human_0 = np.column_stack((x_axis_0, y_axis_0, z_axis_0))
R_robot_0 = R_human_0 @ R_offset

world_targets_0 = absolute_tips_0 - wrist_pos_0
local_targets_0 = world_targets_0 @ R_robot_0

P_human_targets_0 = torch.tensor(local_targets_0, dtype=torch.float32)
P_robot_targets_0 = torch.matmul(M_heuristic, P_human_targets_0)
numpy_targets_0 = P_robot_targets_0.numpy().astype(np.float32)

for _ in range(10):
    retargeting.retarget(numpy_targets_0)

# =======================================================================
# Phase 4: Frame-by-Frame Optimization Loop
# =======================================================================
print("\n=== 4. Starting Frame-by-Frame Optimization ===")

# Container to store the calculated joint angles for all frames
trajectory_list = []

# [vector]
for i in tqdm(range(n_frames), desc="Optimizing Trajectory"):
    absolute_tips = verts_3d[i, tip_vertex_indices]
    human_wrist_pos = joints_3d[i, 21]

    M = joints_3d[i, 43]
    I = joints_3d[i, 40]

    y_axis = (M - human_wrist_pos) / np.linalg.norm(M - human_wrist_pos)
    temp_x = (I - M) / np.linalg.norm(I - M)
    z_axis = np.cross(temp_x, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    R_human = np.column_stack((x_axis, y_axis, z_axis))
    R_robot = R_human @ R_offset

    T_robot_4x4 = np.eye(4)
    T_robot_4x4[:3, :3] = R_robot
    rx, ry, rz = tf.euler_from_matrix(T_robot_4x4, axes='sxyz')

    world_targets = absolute_tips - human_wrist_pos
    local_targets = world_targets @ R_robot

    P_human_targets = torch.tensor(local_targets, dtype=torch.float32)
    
    P_robot_targets = torch.matmul(M_heuristic, P_human_targets)

    numpy_targets = P_robot_targets.numpy().astype(np.float32)
    finger_qpos = retargeting.retarget(numpy_targets)
    
    robot_qpos_22 = np.zeros(22)
    robot_qpos_22[0:3] = human_wrist_pos     
    robot_qpos_22[3:6] = [rx, ry, rz]     
    robot_qpos_22[6:22] = finger_qpos
    trajectory_list.append(robot_qpos_22)
trajectory_array = np.stack(trajectory_list, axis=0)

# =======================================================================
# Phase 4: Save Dataset
# =======================================================================
np.save(OUTPUT_FILE, trajectory_array)
print(f"\n🎉 SUCCESS! Trajectory saved to '{OUTPUT_FILE}' 🎉")
print(f"Final Dataset Shape: {trajectory_array.shape} (Frames, DoF)")