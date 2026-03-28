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

obj_name = data.get('obj_name', 'Unknown Object')
resolution = data.get('resolution', 64)
grid_dim = data.get('grid_dim_meters', 0)

print(f"-> Object: {obj_name}")
print(f"-> Resolution: {resolution}x{resolution}x{resolution}")
print(f"-> Grid Size: {grid_dim:.4f} m")

# ==========================================
# 📊 시각화 1: 2D 단면도 (Heatmap)
# ==========================================
print("\n-> 1. 2D 단면도(Z축 중간 지점)를 렌더링합니다. (창을 닫으면 다음으로 넘어갑니다)")

# Z축의 정중앙 단면 가져오기
mid_z = resolution // 2
slice_2d = sdf_numpy[:, :, mid_z]

plt.figure(figsize=(8, 6))
# coolwarm 컬러맵: 파란색(음수, 물체 내부), 빨간색(양수, 물체 외부)
img = plt.imshow(slice_2d, cmap='coolwarm', origin='lower')
plt.colorbar(img, label='Signed Distance (meters)')

# SDF 값이 정확히 0인 표면을 검은색 등고선으로 표시
plt.contour(slice_2d, levels=[0.0], colors='black', linewidths=2)

plt.title(f"SDF Cross-Section (Z={mid_z}) - Black line is Surface (0.0)")
plt.xlabel("X axis (grid index)")
plt.ylabel("Y axis (grid index)")
plt.tight_layout()
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