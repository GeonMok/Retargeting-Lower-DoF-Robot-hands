# evaluate_trajectory.py
import os
import glob
import csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import re
import numpy as np
import smplx
import cv2
import trimesh
import trimesh.transformations as tf
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Use the RobotWrapper from dex_retargeting to compute Forward Kinematics
from dex_retargeting.robot_wrapper import RobotWrapper

# =======================================================================
# 1. Base Directories & Configurations
# =======================================================================
BASE_DATASET_DIR = "C:/4-1/KIAT/GRAB/dataset_unzipped/grab"
BASE_MESH_DIR = "C:/4-1/KIAT/GRAB/dataset_unzipped/tools/object_meshes/contact_meshes"
SMPLX_MODEL_PATH = "C:/4-1/KIAT/models"
URDF_PATH = "C:/4-1/KIAT/allegro_hand_description/allegro_hand_description/urdf/allegro_hand_description_right.urdf"

TRAJECTORY_DIR = "C:/4-1/kiat/codes/baseline_outputs"
CSV_OUTPUT_FILE = "evaluation_results.csv"

# =======================================================================
# 2. Helper Functions
# =======================================================================
def parse_filename(filename):
    """
    Extracts subject, object, and action from the generated filename.
    Example: 'allegro_trajectory_s1_apple_eat_1_advanced.npy' -> ('originalloss', 's1', 'apple', 'eat_1')
    """
    pattern = r'allegro_([a-zA-Z0-9_]+)_(s\d+)_([a-zA-Z]+)_([a-zA-Z0-9_]+)_(drop|simple|advanced)\.npy'
    
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match the expected format.")
    
    loss_type, subject, obj_name, action, mode = match.groups()
    
    # Construct exact paths based on parsed info
    npz_path = os.path.join(BASE_DATASET_DIR, subject, f"{obj_name}_{action}.npz")
    ply_path = os.path.join(BASE_MESH_DIR, f"{obj_name}.ply")
    
    return npz_path, ply_path, subject, obj_name, action, mode, loss_type

def compute_smoothness(trajectory: np.ndarray):
    """
    Calculates the Manipulation Smoothness (Mean Absolute Jerk).
    Lower value means smoother movement.
    """
    finger_qpos = trajectory[:, 6:22] # Extract 16 finger joints
    velocity = np.diff(finger_qpos, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    return np.mean(np.abs(jerk))

# =======================================================================
# 3. Main Evaluation Routine
# =======================================================================
def evaluate(trajectory_path):
    filename = os.path.basename(trajectory_path)
    npz_path, ply_path, subject, obj_name, action, mode, loss_type = parse_filename(filename)

    # Load Trajectory
    trajectory = np.load(trajectory_path) # Shape: (N, 22)
    n_frames = trajectory.shape[0]

    print(f"\nGT Trajectory file: {npz_path}, Object Mesh file: {ply_path}\n")
    
    # Load GRAB Data
    data = np.load(npz_path, allow_pickle=True)
    human_gender = str(data['gender'])
    rhand_params = torch.tensor(data['rhand'].item()['params']['fullpose'], dtype=torch.float32)
    body_transl = torch.tensor(data['body'].item()['params']['transl'], dtype=torch.float32)
    body_orient = torch.tensor(data['body'].item()['params']['global_orient'], dtype=torch.float32)
    body_pose = torch.tensor(data['body'].item()['params']['body_pose'], dtype=torch.float32)
    
    obj_transl = data['object'].item()['params']['transl']
    obj_orient = data['object'].item()['params']['global_orient']

    # Initialize SMPL-X
    hand_model = smplx.create(SMPLX_MODEL_PATH, model_type='smplx', gender=human_gender, 
                              use_pca=False, flat_hand_mean=True, batch_size=n_frames)
    
    output = hand_model(transl=body_transl, global_orient=body_orient, 
                        body_pose=body_pose, right_hand_pose=rhand_params, return_verts=True)
    human_verts = output.vertices.detach().numpy() # (N, 10475, 3)

    # Initialize Robot Wrapper (for Forward Kinematics)
    robot = RobotWrapper(URDF_PATH)
    tip_link_names = ['link_15_tip', 'link_3_tip', 'link_7_tip', 'link_11_tip']
    tip_link_indices = [robot.get_link_index(name) for name in tip_link_names]
    
    # Get all movable joint indices to check for whole-finger penetration
    all_joint_indices = list(range(robot.dof))

    # Load Object Mesh
    obj_mesh = trimesh.load(ply_path)

    # SMPL-X tip vertex indices (Thumb, Index, Middle, Ring, Pinky)
    human_tip_indices = [8079, 7669, 7794, 7905, 8022]

    # Metrics Containers
    metrics = {
        "l2_center_dist": [],
        # "penetration_depth": [],
        # "contact_points": [],
        "self_collision_frames": 0
    }

    all_r_joints_obj_space = []

    for i in tqdm(range(n_frames), desc="   -> 1/2 Kinematics", leave=False):        
        # --- 1. Get Human Fingertips ---
        h_tips = human_verts[i, human_tip_indices] # (5, 3)
        h_center = np.mean(h_tips, axis=0)         # (3,)

        # --- 2. Get Robot Fingertips & Joints (World Frame) ---
        qpos_wrist_pos = trajectory[i, 0:3]
        qpos_wrist_euler = trajectory[i, 3:6]
        qpos_fingers = trajectory[i, 6:22]

        # Calculate Robot Base Transform (T_robot)
        T_robot = tf.euler_matrix(*qpos_wrist_euler, axes='sxyz')
        T_robot[:3, 3] = qpos_wrist_pos

        # Compute Local FK for fingers
        robot.compute_forward_kinematics(qpos_fingers)
        
        # Get World Poses for fingertips
        r_tips_world = []
        for idx in tip_link_indices:
            r_tips_world.append((T_robot @ robot.get_link_pose(idx))[:3, 3])
        r_tips_world = np.array(r_tips_world) # (4, 3)
        r_center = np.mean(r_tips_world, axis=0) # (3,)

        # Get World Poses for ALL joints (for penetration testing)
        r_joints_world = []
        for idx in all_joint_indices:
            r_joints_world.append((T_robot @ robot.get_link_pose(idx))[:3, 3])
        r_joints_world = np.array(r_joints_world) # (N, 3) N: # of joints
        
        metrics["l2_center_dist"].append(np.linalg.norm(r_center - h_center))

        # --- Metric B: Self-Collision ---
        # if np.any(pdist(r_tips_world) < 0.015):
        #     metrics["self_collision_frames"] += 1

        dist_matrix = squareform(pdist(r_joints_world))
        finger_ids = np.repeat(np.arange(4), 4)
        same_finger_mask = (finger_ids[:, None] == finger_ids[None, :])
        dist_matrix[same_finger_mask] = np.inf
        if np.any(dist_matrix < 0.015):
            metrics["self_collision_frames"] += 1

        # --- 3. Object Transform & Proximity ---
        # Instead of transforming the mesh (slow), we transform robot points to object local space
        T_obj = np.eye(4)
        T_obj[:3, :3], _ = cv2.Rodrigues(obj_orient[i])
        T_obj[:3, 3] = obj_transl[i]
        T_obj_inv = np.linalg.inv(T_obj)

        # Transform robot joints to object space
        r_joints_homogeneous = np.hstack((r_joints_world, np.ones((len(r_joints_world), 1))))
        r_joints_obj_space = (T_obj_inv @ r_joints_homogeneous.T).T[:, :3]

        all_r_joints_obj_space.append(r_joints_obj_space)
        
    batched_joints = np.vstack(all_r_joints_obj_space) 
    chunk_size = 160 
    total_points = len(batched_joints)
    
    distances_list = []
    is_inside_list = []
    
    for start_idx in tqdm(range(0, total_points, chunk_size), desc="   -> 2/2 Collisions", leave=False):
        end_idx = min(start_idx + chunk_size, total_points)
        chunk_points = batched_joints[start_idx:end_idx]
        
        _, chunk_dists, _ = trimesh.proximity.closest_point(obj_mesh, chunk_points)
        chunk_inside = obj_mesh.contains(chunk_points)
        
        distances_list.append(chunk_dists)
        is_inside_list.append(chunk_inside)
        
    distances = np.concatenate(distances_list).reshape(n_frames, 16)
    is_inside = np.concatenate(is_inside_list).reshape(n_frames, 16)
    
    CONTACT_THRESHOLD = 0.01 
    
    penetrated_per_frame = np.sum(is_inside, axis=1) 
    contact_per_frame = np.sum((distances < CONTACT_THRESHOLD) & (~is_inside), axis=1)

    return {
        "Filename": filename,
        "Loss_Type": loss_type,
        "Subject": subject,
        "Object": obj_name,
        "Action": action,
        "Mode": mode,
        "L2_Error_mm": round(np.mean(metrics["l2_center_dist"]) * 1000, 2),
        "Jerk": round(compute_smoothness(trajectory), 6),
        "Contact_Pts": round(float(np.mean(contact_per_frame)), 2),
        "Penetration": round(float(np.mean(penetrated_per_frame)), 2),
        "Self_Collision_%": round((metrics["self_collision_frames"] / n_frames) * 100, 2)
    }

if __name__ == "__main__":
    target_files = glob.glob(os.path.join(TRAJECTORY_DIR, "*.npy"))
    print(f"🔍 Found {len(target_files)} trajectory files to evaluate.\n")
    
    all_results = []
    
    for idx, filepath in enumerate(target_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(target_files)}] Evaluating: {filename}")
        
        try:
            result = evaluate(filepath)
            all_results.append(result)
        except Exception as e:
            print(f"🚨 Error processing {filename}: {e}")
    
    if all_results:
        keys = all_results[0].keys()
        with open(CSV_OUTPUT_FILE, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
            
        print(f"\n🎉 ALL DONE! Results successfully saved to: {CSV_OUTPUT_FILE}")