# bake_sdf.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_sdf, get_surface_point_cloud

def bake_general_object_sdf(mesh_path, output_dir, resolution=64, padding_rate=0.2):
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
    output_path = os.path.join(output_dir, f"{obj_name}_sdf_res{resolution}.pt")
    
    print(f"\n📢 Baking SDF for object: '{obj_name}'...")
    print(f"-> Loading mesh from: {mesh_path}")
    
    mesh = trimesh.load(mesh_path)
    print(f"-> Is the mesh perfectly watertight? {mesh.is_watertight}")
    
    # Calulate original spatial properties
    old_center = mesh.bounds.mean(axis=0) # Object Local Center
    old_extents = mesh.extents # Dimensions of the bounding box (width, height, depth)
    old_scale = old_extents.max()
    
    # Normalize the mesh
    mesh.vertices -= old_center
    mesh.vertices /= old_scale

    # Define the Grid Bounds
    limit = 0.5 + padding_rate
    grid_coords = np.linspace(-limit, limit, resolution)
    x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    
    print(f"-> Computing Signed Distances for {len(points)} grid points...")

    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    
    # Compute SDF
    try:
        # cloud = get_surface_point_cloud(mesh)
        # sdf_values = cloud.get_sdf_in_batches(points)
        sdf_values = mesh_to_sdf(mesh, points, sign_method='depth')
    except Exception as e:
        print("🚨 Error during SDF computation:", e)
        print("Falling back to 'tree' method")
        sdf_values = mesh_to_sdf(mesh, points, sign_method='tree')

    # Denormaalize the SDF values to original scale
    sdf_values_original_scale = sdf_values * old_scale
    
    # Format for PyTorch 
    sdf_grid_tensor = torch.from_numpy(sdf_values_original_scale.reshape(1, resolution, resolution, resolution)).float()
    grid_dim_meters = (2 * limit) * old_scale # The physical size of the grid in meters after scaling back
    
    data_to_save = {
        'sdf_grid': sdf_grid_tensor, 
        'obj_name': obj_name,
        'resolution': resolution,
        'grid_dim_meters': grid_dim_meters, 
        'padding_rate': padding_rate,
        'old_center': old_center,
        'old_scale': old_scale
    }
    
    torch.save(data_to_save, output_path)
    print(f"\n🎉 SUCCESS! SDF baked to: '{output_path}' 🎉")
    print(f"   Grid Extents: {grid_dim_meters:.4f} meters (Cube)")

if __name__ == "__main__":
    GRAB_TOOLS_MESH_DIR = "C:/4-1/KIAT/GRAB/dataset_unzipped/tools/object_meshes/contact_meshes"
    OUTPUT_DIR = "./baked_sdfs"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    obj_list = ["airplane", "apple", "camera", "cup", "alarmclock", "banana", "binoculars", "bowl", "cubelarge", "cubemedium", "cubesmall", "cylinderlarge", "cylindermedium", "cylindersmall", "doorknob"]
    for obj_name in obj_list:
        mesh_path = os.path.join(GRAB_TOOLS_MESH_DIR, f"{obj_name}.ply")
        bake_general_object_sdf(mesh_path, OUTPUT_DIR, resolution=64)