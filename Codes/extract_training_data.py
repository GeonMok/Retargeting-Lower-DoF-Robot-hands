# extract_training_data.py (로컬 PC에서 실행)
import os
import glob
import torch
import smplx
import numpy as np
from tqdm import tqdm

DATASET_DIR = "C:/4-1/kiat/Dataset/trajectory"
MODEL_PATH = "C:/4-1/KIAT/models"

npz_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
all_human_tips_local = []

print("Extracting human fingertip coordinates...")
for npz_path in tqdm(npz_files):
    data = np.load(npz_path, allow_pickle=True)
    gender = str(data['gender'])
    n_frames = data['n_frames']
    
    hand_model = smplx.create(MODEL_PATH, model_type='smplx', gender=gender, 
                              use_pca=False, flat_hand_mean=True, batch_size=n_frames)
    
    # 파라미터 텐서 변환 (생략: 기존 코드와 동일하게 transl, global_orient 등 준비)
    transl = torch.tensor(data['body'].item()['params']['transl'], dtype=torch.float32)
    global_orient = torch.tensor(data['body'].item()['params']['global_orient'], dtype=torch.float32)
    body_pose = torch.tensor(data['body'].item()['params']['body_pose'], dtype=torch.float32)
    right_hand_pose = torch.tensor(data['rhand'].item()['params']['fullpose'], dtype=torch.float32)

    output = hand_model(transl=transl, global_orient=global_orient, 
                        body_pose=body_pose, right_hand_pose=right_hand_pose, return_verts=True)
    
    # 1. 사람 손목(Wrist)과 손끝(Fingertips) 3D 좌표 추출
    human_wrist = output.joints[:, 21, :] # (N, 3)
    tip_indices = [8079, 7669, 7794, 7905, 8022] 
    absolute_tips = output.vertices[:, tip_indices, :] # (N, 5, 3)
    
    # 2. 손목을 (0,0,0)으로 하는 Local 좌표계로 변환! (네트워크 학습이 훨씬 안정적임)
    local_tips = absolute_tips - human_wrist.unsqueeze(1) # (N, 5, 3)
    
    all_human_tips_local.append(local_tips.detach().numpy())

# 수만 프레임의 데이터를 하나의 가벼운 배열로 합치기
dataset_array = np.concatenate(all_human_tips_local, axis=0) # (Total_Frames, 5, 3)
np.save("human_tips_dataset.npy", dataset_array)
print(f"🎉 Saved! Shape: {dataset_array.shape}, Size: {dataset_array.nbytes / 1024 / 1024:.2f} MB")