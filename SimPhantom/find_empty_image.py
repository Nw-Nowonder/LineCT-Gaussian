import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def analyze_empty_projections(nii_path):
    nii_data = nib.load(nii_path)
    proj_data = nii_data.get_fdata().astype(np.float32)
    n_projs = proj_data.shape[2]
    
    intensities = []
    for i in range(n_projs):
        img = proj_data[:, :, i]
        intensities.append(np.mean(img))
    
    intensities = np.array(intensities)
    
    plt.figure(figsize=(12, 4))
    plt.plot(intensities, label="Image Mean Intensity")
    plt.axhline(y=np.max(intensities)*0.05, color='r', linestyle='--', label="5% Threshold")
    plt.title("Projection Intensity along Linear Track")
    plt.xlabel("Projection Index (0 to 970)")
    plt.ylabel("Mean Pixel Value")
    plt.legend()
    plt.show()

    threshold = np.max(intensities) * 0.05
    valid_indices = np.argwhere(intensities > threshold).flatten()
    
    print(f"总图片数: {n_projs}")
    print(f"包含有效信息的图片数: {len(valid_indices)}")
    print(f"建议保留的索引范围: 从 {valid_indices[0]} 到 {valid_indices[-1]}")

if __name__ == "__main__":
    analyze_empty_projections("./SimPhantom./SimPhantom.nii")