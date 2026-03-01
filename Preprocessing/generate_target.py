import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import smplx
import numpy as np

MODEL_PATH = "C:/4-1/KIAT/models"  
NPZ_FILE = 'C:/4-1/KIAT/GRAB/dataset_unzipped/grab/s1/banana_eat_1.npz'

def get_human_keypoints_from_grab(npz_path, model_path):

    data = np.load(npz_path, allow_pickle=True)
    rhand_params = data['rhand'].item()['params']
    
    n_frames = data['n_frames']
    gender = str(data['gender'])
    hand_model = smplx.create(model_path, 
                              model_type='smplx',
                              gender=gender, 
                              use_pca=False,
                              flat_hand_mean=True,
                              batch_size=n_frames)

    fullpose = torch.tensor(rhand_params['fullpose'], dtype=torch.float32)
    global_orient = torch.tensor(rhand_params['global_orient'], dtype=torch.float32)
    transl = torch.tensor(rhand_params['transl'], dtype=torch.float32)

    output = hand_model(
        global_orient=global_orient,
        right_hand_pose=fullpose,
        transl=transl,
        return_verts=True # no mesh vertices needed, but we want joints
    )

    joints_3d = output.joints.detach().numpy() # (N, 127, 3): frames, joints, XYZ
    
    return joints_3d, data

joints, raw_data = get_human_keypoints_from_grab(NPZ_FILE, MODEL_PATH)

print(f"=== Data transformation completed ===")
print(f"Transformed shape: {joints.shape}") 

wrist_pos = joints[0, 0] # Wrist
finger_tips_indices = [4, 8, 12, 16, 20] 

# Relative Pose
relative_tips = joints[0, finger_tips_indices] - wrist_pos

print("\n=== Target Candidates (Relative) ===")
for i, name in zip(range(5), ["Thumb", "Index", "Middle", "Ring", "Pinky"]):
    print(f"{name}: {relative_tips[i]}")