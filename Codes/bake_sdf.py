# bake_sdf.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_sdf
from tqdm import tqdm
import argparse

def bake_general_object_sdf(mesh_path, output_dir, resolution=64, padding_rate=0.2):
    """
    어떤 일반적인 물체 메쉬(.ply)든 입력받아 SDF 그리드 텐서로 구워냅니다.
    로봇 손가락이 접근하는 것을 미리 감지하기 위해 충분한 Padding을 둡니다.
    """
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
    output_path = os.path.join(output_dir, f"{obj_name}_sdf_res{resolution}.pt")
    
    print(f"\n📢 Baking SDF for object: '{obj_name}'...")
    print(f"-> Loading mesh from: {mesh_path}")
    
    # 1. 메쉬 로드 및 정규화(SDF 연산의 안정성을 위해)
    mesh = trimesh.load(mesh_path)
    
    # 원본 바운딩 박스 정보 저장 (나중에 복원을 위해 필수!)
    old_center = mesh.bounds.mean(axis=0)
    old_extents = mesh.extents
    old_scale = old_extents.max()
    
    # 메쉬를 원점으로 가져오고, 긴 축이 1이 되도록 스케일링 (mesh_to_sdf 요구사항)
    mesh.vertices -= old_center
    mesh.vertices /= old_scale
    
    # 2. SDF 그리드 좌표 생성 (Padding 포함)
    # mesh_to_sdf는 기본적으로 -0.5 ~ 0.5 범위 내의 메쉬를 가정합니다.
    # 기구학적 마진(로봇 손이 다가올 때 미리 감지)을 위해 0.5 + padding_rate 범위까지 굽습니다.
    limit = 0.5 + padding_rate
    
    # -limit ~ limit 사이의 해상도 점들을 가진 정육면체 그리드 생성
    voxel_origin = np.array([-limit, -limit, -limit])
    voxel_size = (2 * limit) / (resolution - 1)
    
    # 3D 그리드 좌표 행렬 만들기 (num_points, 3)
    grid_coords = np.linspace(-limit, limit, resolution)
    x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    
    # 3. Signed Distance 계산 (여기가 시간이 조금 걸립니다, tqdm 활용)
    print(f"-> Computing Signed Distances for {len(points)} grid points...")
    print("   (This might take a few minutes for complex meshes)")
    
    # mesh_to_sdf 함수 호출 (sign_method='normal'이 그나마 정확함)
    sdf_values = mesh_to_sdf(mesh, points, sign_method='normal')
    
    # 4. 스케일 복원 및 텐서 변환
    # mesh_to_sdf의 결과는 정규화된 스케일 기준이므로, 원본 스케일을 곱해줍니다.
    sdf_values_original_scale = sdf_values * old_scale
    
    # 데이터를 (1, 해상도, 해상도, 해상도) 형태의 PyTorch 텐서로 변환
    # PyTorch의 grid_sample 함수는 (Batch, Depth, Height, Width) 형식을 요구함
    sdf_grid_tensor = torch.from_numpy(sdf_values_original_scale.reshape(1, resolution, resolution, resolution)).float()
    
    # 5. 메타데이터와 함께 저장 (최적화 루프에서 복원을 위해 필수!)
    # grid_dim_meters: SDF 그리드 한 변의 실제 길이 (미터 단위)
    grid_dim_meters = (2 * limit) * old_scale
    
    data_to_save = {
        'sdf_grid': sdf_grid_tensor, # 3D SDF 텐서
        'obj_name': obj_name,
        'resolution': resolution,
        'grid_dim_meters': grid_dim_meters, # 그리드 전체 크기(m)
        'padding_rate': padding_rate
    }
    
    torch.save(data_to_save, output_path)
    print(f"\n🎉 SUCCESS! SDF baked to: '{output_path}' 🎉")
    print(f"   Grid Extents: {grid_dim_meters:.4f} meters (Cube)")

if __name__ == "__main__":
    # GRAB 데이터셋 구조에 맞게 일반적인 사용 예시 작성
    GRAB_TOOLS_MESH_DIR = "C:/4-1/KIAT/GRAB/dataset_unzipped/tools/object_meshes/contact_meshes"
    OUTPUT_DIR = "./baked_sdfs"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 예시: 사과(Apple) 굽기
    # apple_path = os.path.join(GRAB_TOOLS_MESH_DIR, "apple.ply")
    # bake_general_object_sdf(apple_path, OUTPUT_DIR, resolution=64)
    
    # 예시: 다른 일반적인 물체(예: 카메라) 굽기 - 주석 해제 후 사용
    camera_path = os.path.join(GRAB_TOOLS_MESH_DIR, "camera.ply")
    bake_general_object_sdf(camera_path, OUTPUT_DIR, resolution=64)