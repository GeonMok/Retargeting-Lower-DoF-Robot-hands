import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from skimage.measure import marching_cubes
import os

# 1. 경로 설정 (구워진 파일 경로)
SDF_FILE_PATH = "C:/4-1/KIAT/Codes/baked_sdfs/camera_sdf_res64.pt"

if not os.path.exists(SDF_FILE_PATH):
    print(f"🚨 파일을 찾을 수 없습니다: {SDF_FILE_PATH}")
    exit()

print(f"-> Loading baked SDF from: {SDF_FILE_PATH}")
data = torch.load(SDF_FILE_PATH)
sdf_tensor = data['sdf_grid']  # Shape: (1, 64, 64, 64)
sdf_numpy = sdf_tensor.squeeze().numpy()  # Shape: (64, 64, 64)
print(f"SDF Data Range: Min = {sdf_numpy.min():.4f}, Max = {sdf_numpy.max():.4f}")

obj_name = data.get('obj_name', 'Unknown Object')
resolution = data.get('resolution', 64)
grid_dim = data.get('grid_dim_meters', 0)

print(f"-> Object: {obj_name}")
print(f"-> Resolution: {resolution}x{resolution}x{resolution}")
print(f"-> Grid Size: {grid_dim:.4f} m")

# ==========================================
# 📊 시각화 1: 2D 단면도 (Heatmap)
# ==========================================
print("\n-> 1. 2D 단면도를 렌더링합니다. (창을 닫으면 다음으로 넘어갑니다)")

# 3개의 다른 Z 높이 지정 (ex: Z=16, Z=32, Z=48)
z_slices = [resolution // 3, resolution // 2, resolution // 3 * 2]  

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
max_dist = np.max(np.abs(sdf_numpy)) # Make colormap symmetrical

for i, z_idx in enumerate(z_slices):
    slice_2d = sdf_numpy[:, :, z_idx]
    
    # 렌더링
    ax = axes[i]
    img = ax.imshow(slice_2d, cmap='coolwarm', origin='lower', vmin=-max_dist, vmax=max_dist)
    
    # 등고선 (표면)
    ax.contour(slice_2d, levels=[0.0], colors='black', linewidths=2)
    
    ax.set_title(f"Slice at Z={z_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# 공통 컬러바 추가
cbar = fig.colorbar(img, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('Signed Distance (meters)')

plt.suptitle(f"SDF Cross-Sections for {obj_name}")
plt.show()
# ==========================================
# 🍎 시각화 2: 3D 표면 복원 (Marching Cubes)
# ==========================================
print("\n-> 2. SDF 값이 0인 지점들을 연결해 3D 메쉬로 복원합니다...")

try:
    # Marching Cubes 알고리즘: 3D 볼륨 데이터에서 특정 값(level=0.0)의 표면을 추출합니다.
    verts, faces, normals, values = marching_cubes(sdf_numpy, level=0.0)
    
    # 3D 메쉬 생성
    reconstructed_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # 색상 입히기 (녹색 사과 느낌)
    reconstructed_mesh.visual.vertex_colors = [100, 255, 100, 255]
    
    print("-> 3D 뷰어를 띄웁니다! 마우스로 돌려보세요.")
    reconstructed_mesh.show(caption=f"Reconstructed Surface from SDF ({obj_name})")

except ValueError as e:
    print(f"\n🚨 3D 복원 실패: {e}")
    print("SDF 그리드 안에 0 값(표면)이 온전히 포함되지 않았을 수 있습니다. padding을 늘려보세요.")