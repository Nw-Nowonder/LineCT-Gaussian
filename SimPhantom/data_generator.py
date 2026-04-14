import json
import numpy as np
import os
import nibabel as nib
# from tqdm import trange

def generate_linear_ct_metadata(output_dir, object_scale=50.0):
    # make meta_data.json

    # configuration parameters
    os.makedirs(output_dir, exist_ok=True)
    
    dso_mm = 600.0
    dsd_mm = 1000.0
    
    n_detector = [280, 350]
    pixel_size_mm = 10.0
    s_detector_mm = [n_detector[0] * pixel_size_mm, n_detector[1] * pixel_size_mm]
    
    n_voxel = [80, 300, 300]
    voxel_size_mm = 4.0
    s_voxel_mm = [n_voxel[0] * voxel_size_mm, n_voxel[1] * voxel_size_mm, n_voxel[2] * voxel_size_mm]
    
    n_projs = 971
    step_mm = 4.0
    
    # switch to physical units and apply object scale
    def scale_val(val_mm):
        return (val_mm / 1000.0) * object_scale

    dso = scale_val(dso_mm)
    dsd = scale_val(dsd_mm)
    s_detector = [scale_val(s_detector_mm[0]), scale_val(s_detector_mm[1])]
    s_voxel = [scale_val(s_voxel_mm[0]), scale_val(s_voxel_mm[1]), scale_val(s_voxel_mm[2])]
    
    off_origin_mm = [0.0, 0.0, 0.0]
    off_origin = [scale_val(off_origin_mm[0]), scale_val(off_origin_mm[1]), scale_val(off_origin_mm[2])]
    
    bbox = [
        [off_origin[0] - s_voxel[0]/2, off_origin[1] - s_voxel[1]/2, off_origin[2] - s_voxel[2]/2],
        [off_origin[0] + s_voxel[0]/2, off_origin[1] + s_voxel[1]/2, off_origin[2] + s_voxel[2]/2]
    ]

    # 3. make proj_train list
    total_travel_mm = (n_projs - 1) * step_mm
    start_pos_mm = -total_travel_mm / 2.0
    
    proj_train = []
    proj_test = []
    for i in range(300,671):
        current_x_mm = start_pos_mm + i * step_mm
        current_x_scaled = scale_val(current_x_mm)
        
        file_name = f"{i:04d}.npy"
        
        if i % 10 == 5:  # choose every 10th projection for testing
            proj_test.append({
                "translation_x": current_x_scaled, 
                "file_path": f"proj_test/{file_name}",
            })
        else:
            proj_train.append({
                "translation_x": current_x_scaled, 
                "file_path": f"proj_train/{file_name}",
            })

    # 4. make meta_data dict
    meta_data = {
        "scanner": {
            "mode": "cone",
            "DSD": dsd,
            "DSO": dso,
            "nDetector": n_detector,
            "sDetector": s_detector,
            "nVoxel": n_voxel,
            "sVoxel": s_voxel,
            "offOrigin": off_origin,
            "offDetector": [0.0, 0.0],
            "accuracy": 0.5,
            "totaltrans": total_travel_mm,
            "starttrans": start_pos_mm,
            "noise": True, 
            "filter": None
        },
        "radius": 1.0,
        "bbox": bbox,
        "vol": "vol_gt.npy",
        "proj_train": proj_train,
        "proj_test": proj_test,
    }

    output_path = os.path.join(output_dir, "meta_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)
    
    print(f"meta_data.json 已成功生成至: {output_path}")

def generate_linear_ct_training_data(nii_path, output_dir):

    nii_data = nib.load(nii_path)
    proj_data = nii_data.get_fdata().astype(np.float32) 
    proj_data = np.transpose(proj_data, (1, 0, 2))
    print(proj_data.shape)   #输出结果是(280, 350, 971)

    train_dir = os.path.join(output_dir, "proj_train")
    test_dir = os.path.join(output_dir, "proj_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    n_projs = proj_data.shape[2]  
    print(f"共检测到 {n_projs} 张投影")

    for i in range(300,671):
        single_proj = proj_data[:, :, i]
        
        file_name = f"{i:04d}.npy" 
        
        if i % 10 == 5:
            save_path = os.path.join(test_dir, file_name)
        else:
            save_path = os.path.join(train_dir, file_name)
            
        np.save(save_path, single_proj)
        
    print(f"投影图像保存在 {output_dir}")

def generate_linear_ct_model_data(nii_path, output_dir):
    nii_data = nib.load(nii_path)
    vol_data = nii_data.get_fdata().astype(np.float32)
    vol_data = np.transpose(vol_data, (2, 0, 1))
    print(vol_data.shape)   #输出结果是(80, 300, 300)

    vol_save_path = os.path.join(output_dir, "vol_gt.npy")
    np.save(vol_save_path, vol_data)
    
    print(f"体积数据已保存至: {vol_save_path}")

if __name__ == "__main__":
    generate_linear_ct_metadata(output_dir="./SimPhantom/data", object_scale=50.0)
    generate_linear_ct_training_data(nii_path="./SimPhantom/SimPhantom.nii", output_dir="./SimPhantom/data")
    generate_linear_ct_model_data(nii_path="./SimPhantom/SimPhantom.nii.gz", output_dir="./SimPhantom/data")