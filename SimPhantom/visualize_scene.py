import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

file_path = "./SimPhantom/SimPhantom.nii"

nii_img = nib.load(file_path)
image_data = nii_img.get_fdata()

n_proj = image_data.shape[2]
step_mm = 4.0

start_pos = - ((n_proj - 1) / 2.0) * step_mm
all_offsets = np.array([start_pos + i * step_mm for i in range(n_proj)])

n_samples = 5
sample_indices = np.linspace(300, 671, n_samples).astype(int)
sample_offsets = all_offsets[sample_indices]

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, n_samples, height_ratios=[4, 1], hspace=0.1)

for i, idx in enumerate(sample_indices):
    ax_img = fig.add_subplot(gs[0, i])
    
    proj_img = image_data[:, :, idx]
    
    ax_img.imshow(proj_img, cmap='bone', aspect='auto')
    
    ax_img.set_title(f"Proj {idx}\n{sample_offsets[i]:.1f} mm", fontsize=12, fontweight='bold', color='darkblue')
    ax_img.axis('off') 

ax_line = fig.add_subplot(gs[1, :])


ax_line.axhline(0, color='black', linewidth=1.5)

ax_line.scatter(all_offsets, np.zeros_like(all_offsets), color='lightgray', s=5, label='All 971 Projections')

ax_line.scatter(sample_offsets, np.zeros_like(sample_offsets), color='red', s=80, zorder=5, label='Sampled Projections')

for offset in sample_offsets:
    ax_line.axvline(offset, ymin=0.5, ymax=1.5, color='red', linestyle='--', alpha=0.5, clip_on=False)

ax_line.set_yticks([])  
ax_line.set_xlabel('X-axis Translation Offset (mm)', fontsize=14, fontweight='bold')
ax_line.set_xlim(all_offsets[0] - 100, all_offsets[-1] + 100)
ax_line.spines['top'].set_visible(False)
ax_line.spines['right'].set_visible(False)
ax_line.spines['left'].set_visible(False)
ax_line.legend(loc='lower right')

fig.suptitle('Linear CT Scanning Process: Projection Images along Translation Axis', fontsize=18, fontweight='bold', y=1.05)

plt.tight_layout()
plt.show()