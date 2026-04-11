import numpy as np
import trimesh
import trimesh.transformations as tf
from yourdfpy import URDF
import os
import cv2
import warnings
import torch # SMPL-X 구동을 위해 필요
import smplx # 인간 손 메쉬 생성을 위해 필요
warnings.filterwarnings('ignore')

# =========================================================================
# Configuration & Absolute Paths
# =========================================================================
print("=== 1. Setting Up Absolute Paths ===")

# 🚨 사용자의 환경에 맞게 경로를 확인해 주세요.
SMPLX_MODEL_PATH = "C:/4-1/KIAT/models"
TRAJECTORY_FILE = "C:/4-1/KIAT/Codes/allegro_trajectory_s1_apple_eat_1_simple.npy" 
APPLE_MESH_FILE = "C:/4-1/KIAT/GRAB/dataset_unzipped/tools/object_meshes/contact_meshes/apple.ply"
GRAB_NPZ_FILE = "C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/apple_eat_1.npz"
URDF_PATH = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf/allegro_hand_description_right.urdf"

ALLEGRO_JOINT_NAMES = [
    'joint_0', 'joint_1', 'joint_2', 'joint_3',       
    'joint_4', 'joint_5', 'joint_6', 'joint_7',       
    'joint_8', 'joint_9', 'joint_10', 'joint_11',     
    'joint_12', 'joint_13', 'joint_14', 'joint_15'    
]

# =========================================================================
# Phase 2: Data Loading & Human Mesh Reconstruction
# =========================================================================
print("\n=== 2. Loading Data & Reconstructing Human Hand ===")

# 2-A. Load Allegro Trajectory
trajectory = np.load(TRAJECTORY_FILE)
n_frames = trajectory.shape[0]

# 2-B. Load Robot & Object Meshes
robot_urdf = URDF.load(URDF_PATH)
apple_mesh = trimesh.load(APPLE_MESH_FILE)
apple_mesh.visual.face_colors = [100, 255, 100, 150] # Semi-transparent green

# Load Original GRAB Data
grab_data = np.load(GRAB_NPZ_FILE, allow_pickle=True)
obj_data = grab_data['object'].item()['params']
obj_transl = obj_data['transl']          
obj_global_orient = obj_data['global_orient'] 

sync_frames = min(n_frames, obj_transl.shape[0])
print(f"-> Total Synced Frames: {sync_frames}")

# 🚨 2-C. 정적 확인을 위한 타겟 프레임 설정 🚨
# 0 : 처음 프레임 (T-pose 상태인 경우가 많음)
# sync_frames - 1 : 마지막 프레임
TARGET_FRAME_IDX = 0 

print(f"-> 🎯 TARGET FRAME SELECTED: {TARGET_FRAME_IDX}")

# SMPL-X Reconstruction (Just for verification parameters)
print("-> Reconstructing Human Hand Mesh from SMPL-X params...")
body_data = grab_data['body'].item()['params']
rhand_data = grab_data['rhand'].item()['params']
gender = str(grab_data['gender'])

# Initialize SMPL-X model (Just 1 batch for the target frame)
human_model = smplx.create(
    SMPLX_MODEL_PATH, model_type='smplx', gender=gender,
    use_pca=False, flat_hand_mean=True, batch_size=1 # Batch size 1
)

# Move target frame parameters to torch tensors
transl_t = torch.tensor(body_data['transl'][TARGET_FRAME_IDX : TARGET_FRAME_IDX+1], dtype=torch.float32)
global_orient_t = torch.tensor(body_data['global_orient'][TARGET_FRAME_IDX : TARGET_FRAME_IDX+1], dtype=torch.float32)
body_pose_t = torch.tensor(body_data['body_pose'][TARGET_FRAME_IDX : TARGET_FRAME_IDX+1], dtype=torch.float32)
right_hand_pose_t = torch.tensor(rhand_data['fullpose'][TARGET_FRAME_IDX : TARGET_FRAME_IDX+1], dtype=torch.float32)

# Run forward pass
output = human_model(
    transl=transl_t, 
    global_orient=global_orient_t, 
    body_pose=body_pose_t, 
    right_hand_pose=right_hand_pose_t, 
    return_verts=True
)
human_verts_static = output.vertices.detach().numpy()[0] # (10475, 3) 절대 좌표

# Create static human hand mesh
human_mesh_static = trimesh.Trimesh(
    vertices=human_verts_static,
    faces=human_model.faces, 
    process=False
)
human_mesh_static.visual.face_colors = [150, 150, 200, 100] # Semi-transparent blue/grey

print("-> Human Hand Reconstruction Complete.")


# =========================================================================
# Phase 3: Scene Construction (Static)
# =========================================================================
print("\n=== 3. Starting Static Scene Construction ===")

static_scene = trimesh.Scene()
static_scene.add_geometry(apple_mesh, node_name='apple_node')

# 🚨 Add the static human hand mesh
static_scene.add_geometry(human_mesh_static, node_name='human_hand_node')

# --- Calculate Static Transforms for Target Frame ---

# 1. Object (Apple) Transform
current_obj_transl = obj_transl[TARGET_FRAME_IDX]
current_obj_orient_aa = obj_global_orient[TARGET_FRAME_IDX] 
current_obj_rot_matrix, _ = cv2.Rodrigues(current_obj_orient_aa)
T_object_transl = tf.translation_matrix(current_obj_transl)
T_object_rot = np.eye(4); T_object_rot[:3, :3] = current_obj_rot_matrix
T_object = np.dot(T_object_transl, T_object_rot)

# 2. Robot (Allegro) Kinematics & Wrist Transform
qpos_static = trajectory[TARGET_FRAME_IDX]
wrist_transl_xyz = qpos_static[0:3]; wrist_rot_rpy = qpos_static[3:6]; finger_angles = qpos_static[6:22]

# Calculate Wrist Matrix
T_wrist_transl = tf.translation_matrix(wrist_transl_xyz)
T_wrist_rot = tf.euler_matrix(wrist_rot_rpy[0], wrist_rot_rpy[1], wrist_rot_rpy[2], axes='sxyz')
T_wrist = np.dot(T_wrist_transl, T_wrist_rot)

# Update robot joint configuration (URDF internal state)
joint_dict_static = {name: angle for name, angle in zip(ALLEGRO_JOINT_NAMES, finger_angles)}
robot_urdf.update_cfg(joint_dict_static) 

# Add robot sub-meshes with applied transforms
for node_name in robot_urdf.scene.graph.nodes_geometry:
    local_transform, geom_name = robot_urdf.scene.graph[node_name]
    geom = robot_urdf.scene.geometry[geom_name]
    geom.visual.face_colors = [255, 100, 100, 200] # Solid red for robot
    
    # Apply wrist transform to the local transform
    final_transform = np.dot(T_wrist, local_transform)
    
    # Add geometry with final world transform immediately
    static_scene.add_geometry(geom, node_name=f"robot_{node_name}", transform=final_transform)

# Phase 3.5: Apply Transforms to Graph
static_scene.graph.update("apple_node", matrix=T_object)

# Phase 3.6: View Setup (흰 화면 방지)
static_scene.set_camera(distance=0.9, center=wrist_transl_xyz)


# =========================================================================
# Phase 4: Launching Static Viewer
# =========================================================================
print(f"\n=== 4. Launching Static Viewer (Frame {TARGET_FRAME_IDX}) ===")
print("-> Solid Red (Robot) vs. Transparent Blue (Human GT)")
print("-> Animation is STOPPED. Rotate with mouse.")

# 🚨 callback 파라미터를 제거하여 애니메이션을 끕니다.
static_scene.show(resolution=(1024, 768))