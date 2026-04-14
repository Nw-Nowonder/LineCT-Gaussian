import os
import json
import numpy as np
import glob

def generate_initial_point_cloud(data_dir, n_points=50000, density_thresh=0.05, density_rescale=0.15):
    # 1. read meta_data.json & load vol_gt.npy
    meta_path = os.path.join(data_dir, "meta_data.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    scanner = meta_data["scanner"]
    s_voxel = np.array(scanner["sVoxel"])
    n_voxel = np.array(scanner["nVoxel"])
    off_origin = np.array(scanner["offOrigin"])
    
    vol_path = os.path.join(data_dir, meta_data["vol"])
    vol_data = np.load(vol_path).astype(np.float32)

    max_val = np.max(vol_data)

    # 2. normalize if max_val > 2.0 (assuming original density values are in the range [0, 3200], we want to scale them down to [0, 1])
    if max_val > 2.0:
        vol_data = vol_data / max_val
        np.save(vol_path, vol_data)
        
        for folder in ["proj_train", "proj_test"]:
            folder_path = os.path.join(data_dir, folder)
            npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
            for img_file in npy_files:
                img = np.load(img_file).astype(np.float32)
                np.save(img_file, img / max_val)
            
    # 3. choose valid points
    density_mask = vol_data > density_thresh
    valid_indices = np.argwhere(density_mask)

    total_valid = valid_indices.shape[0]
    print(f"Total valid points above density threshold: {total_valid}")
    if total_valid < n_points:
        n_points = total_valid

    # 4. randomly sample n_points from valid_indices
    if total_valid > 0:
        random_choice = np.random.choice(total_valid, n_points, replace=False)
        sampled_indices = valid_indices[random_choice]

        d_voxel = s_voxel / n_voxel 
        sampled_positions = sampled_indices * d_voxel - s_voxel / 2 + off_origin

        scene_scale = 2.0 / np.max(s_voxel)
        sampled_positions = sampled_positions * scene_scale
        
        sampled_densities = vol_data[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * density_rescale

        out_pcd = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
        
        out_path = os.path.join(data_dir, "init_data.npy")
        np.save(out_path, out_pcd)
    else:
        print("No valid points found above the density threshold.")

if __name__ == "__main__":
    generate_initial_point_cloud(
        data_dir="./SimPhantom/data", 
        n_points=50000, 
        density_thresh=0.05,  
        density_rescale=0.15
    )