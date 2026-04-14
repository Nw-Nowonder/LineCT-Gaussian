"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

import os

msvc_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if os.path.exists(msvc_path) and msvc_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + msvc_path
# put msvc_path in PATH to avoid "Microsoft Visual C++ 14.0 is required" error when importing pycuda

from glob import glob
import skimage.io as io
import matplotlib.pyplot as plt
import math
import copy
import numpy
import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import SimpleITK as sitk
import openpyxl
from glob import glob
import pycuda.gpuarray as gpuarray
from concurrent.futures import ThreadPoolExecutor

# from mayavi import mlab

mod = SourceModule('''

__global__ void
FL(int *N_angle,int *Nrow_det,int *Ncol_det,float *SincL,float *proj,float *f_fl){
    long idx = (long)(threadIdx.x + blockDim.x * blockIdx.x);
    long idy = (long)(threadIdx.y + blockDim.y * blockIdx.y);
    long id = (long)(idx + idy * blockDim.x * gridDim.x);
    long Na= (long)N_angle[0];
    long Nc=(long)Ncol_det[0];
    long Nr=(long)Nrow_det[0];
    long N_id=(long)(Na*Nc*Nr);
    
    if (id == 0) {
        printf("N_id = %ld\\n", N_id);
    }
    
    if (id>=N_id){
        return;
    }
    long ia=(long)floorf(id/(Nc*Nr));
    long icr=(long)(id % (Nc*Nr));
    long ic=(long)(icr % Nc);
    long ir=(long)floorf(icr/Nc);
    
    long inda,ind1;           
    float f_return=0;
    
    for (inda = 0; inda < Na; inda++) {
        ind1 = ic + ir*Nc + inda*Nc*Nr;
        f_return += SincL[inda+Na-ia]*proj[ind1];
    }
    f_fl[id]=f_return;
}

__global__ void
BP(float *pi,float *R,float *D,float *Sx_init, int *Nz, int *Nx,int *Ny, int *N_angle, int *Nrow_det, int *Ncol_det, float *phi, float *u_coordinate,float *v_coordinate,float *pxx,float *pyy,float *pzz,float *f_fl,float *rec){

    long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    long long idy = threadIdx.y + blockDim.y * blockIdx.y;
    long long id = idx + idy * blockDim.x * gridDim.x;
    long long Na=N_angle[0];
    long long Nr=Nrow_det[0];
    long long Nc=Ncol_det[0];
    long long nz=Nz[0];
    long long ny=Ny[0];
    long long nx=Nx[0];
    
    long long N_id=Nc*nz*ny*nx;
    if (id >= N_id) {
        return;
    }
    
    long long islook = 60;
    long long ivlook = 112;
    long long iulook = 340;
    
    long long izlook = 35;
    long long iylook = 190;
    long long ixlook = 100;
    
    long long ilook = islook*Nc*Nr+ivlook*Nc+iulook;
    if (id == ilook) {
        printf("f_fl[id] = %f\\n", f_fl[id]);
    }
    
    ilook=iulook*nz*ny*nx+izlook*ny*nx+iylook*nx+ixlook;

    long long izyx=id % (nz*ny*nx);
    long long it=int(floorf(id/(nz*ny*nx)));
    long long iyx=izyx % (ny*nx);
    long long iz=int(floorf(izyx/(ny*nx)));
    long long ix=iyx % nx;
    long long iy=int(floorf(iyx/nx));

    float s_asterisk,v_asterisk;
    float f_return=0;
    float px,py,pz,f1,f2;
    long long iv,is,id11,id12,id21,id22;
    
    px = pxx[ix];
    py = pyy[iy];
    pz = pzz[iz];
    
    float pos_t = u_coordinate[it]*R[0]/D[0];
    
    float theta = pi[0]-atan2(pos_t, R[0]);
    s_asterisk = px - pz * pos_t / R[0] - pos_t;
    v_asterisk = py / (R[0] + pz) * D[0];
    
    float vir_y = py / (R[0] + pz) * R[0];
    float bp_weight = 1/sqrt(pos_t * pos_t + vir_y * vir_y + R[0] * R[0]);

    float du = u_coordinate[1] - u_coordinate[0];
    float dv = v_coordinate[1] - v_coordinate[0];
    float ds = phi[1] - phi[0];
    
    iv = int(floorf((v_asterisk-v_coordinate[0])/dv));
    is = int(floorf((Sx_init[0] - s_asterisk - phi[0])/ds));
    
    if (iv < 0){
        f_return = 0;
    }
    else if (iv>=(Nr-1)){
        f_return = 0;
    }
    else if (is>=(Na-1)){
        f_return = 0;
    }
        else if (is<0){
        f_return = 0;
    }
    else {
        id11=it+iv*Nc+is*Nc*Nr;
        id12=it+(iv+1)*Nc+is*Nc*Nr;
        id21=it+iv*Nc+(is+1)*Nc*Nr;
        id22=it+(iv+1)*Nc+(is+1)*Nc*Nr;
        f1=(f_fl[id12]-f_fl[id11])*(v_asterisk-v_coordinate[iv])/dv + f_fl[id11];
        f2=(f_fl[id22]-f_fl[id21])*(v_asterisk-v_coordinate[iv])/dv + f_fl[id21];
        f_return = f1 + (f2-f1)*(Sx_init[0]-s_asterisk-phi[is])/ds;
    }
    
    atomicAdd(&rec[izyx], f_return * bp_weight * du);
}
''')

# atomicAdd(&rec[izts], f_return * U * angle_interval[0]/(4*pi[0]*pi[0]));

tic = time.time()
fl = mod.get_function("FL")
bp = mod.get_function("BP")


recon_path = "E:\\recon\\LineCL_fbp_ball50320_120256768.nii.gz"

ind_SaveRec=1

# 1.导入原始图像
file_path = ".\SimPhantom\SimPhantom.nii.gz"
data = sitk.ReadImage(file_path)
phantom = sitk.GetArrayFromImage(data).astype(np.float32)
del data, file_path
Attenuation_range=1.6
phantom=phantom/phantom.max() * Attenuation_range

scale = int(phantom.shape[0]/9)
plt.figure()
for inda in range(9):
    plt.subplot(3, 3, inda + 1)
    ia = np.int32(inda * scale)
    plt.imshow(phantom[ia, :, :], cmap="gray")
    plt.title('phantom z=' + str(ia))
    plt.axis('off')

toc1 = time.time()
print('phantom time is :', toc1 - tic)

# 2.导入投影数据
tim_procproj0 = time.time()

trans_state = 0

proj_path = ".\SimPhantom\SimPhantom.nii"
data = sitk.ReadImage(proj_path)
proj_cl = sitk.GetArrayFromImage(data).astype(np.float32)
del proj_path,data

tim_procproj1 = time.time()
state_90=0
if state_90 == 1:
    Ncol_det = 200
    indstart=np.int32((proj_cl.shape[2]-Ncol_det)/2)
    proj_cl=proj_cl[:,:,indstart:indstart+Ncol_det]

state_trunc = 0
if state_trunc == 1:
    Nrow_det = 180
    indstart = np.int32((proj_cl.shape[1] - Nrow_det) / 2)
    proj_cl = proj_cl[:, indstart:indstart + Nrow_det, :]

print('load proj time is :', tim_procproj1 - tim_procproj0)
print()

tim_w0 = time.time()
pi = math.pi
N_angle, Nrow_det, Ncol_det = proj_cl.shape
Nrow_det = np.int32(Nrow_det)
Ncol_det = np.int32(Ncol_det)
N_angle = np.int32(N_angle)

plt.figure()
scale=int(N_angle/9)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(proj_cl[i * scale, :, :], cmap="gray")
    plt.title('proj is=' + str(i * scale))
    plt.axis('off')

plt.figure()
indr = 55
plt.imshow(proj_cl[:, indr, :], cmap="gray")
plt.title('proj_cl ir=' + str(indr))

inclination_angle = np.float32(45 * pi / 180)
scale_rec=1
Nz = 80/scale_rec
Ny = Nx = 300/scale_rec
dx = dy = dz = 4*scale_rec
det_pixel = 10
ds=3

R = distance_source_Os = 600
D = distance_source_detector = 1000

bis_z = 0 * dz
bis_ang = 0
R = np.float32(R)
D = np.float32(D)

print('proj size:',N_angle, Nrow_det, Ncol_det)
print('rec size:',Nz, Ny, Nx)
print('dx,dy,dz:', dx, dy, dz)
print('det_pixel:', det_pixel)
print('R:', R)
print('D:', D)

angle_value=math.atan2(det_pixel*Ncol_det/2,D)
print('angle_value:', np.rad2deg(angle_value*2))

length = dx * Nx
width = dy * Ny
height = dz * Nz

Sx_init = np.float32(ds*(N_angle-1)/2)
phi = np.linspace(0, Sx_init*2, N_angle).astype(numpy.float32)

# Sx_init=np.float32(det_pixel*Ncol_det*0.7*R/D)
# phi = np.linspace(0, Sx_init*2, N_angle).astype(numpy.float32)
# ds=np.float32(Sx_init*2/(N_angle-1))

print('ds:', ds)

detector_u0 = (-det_pixel * Ncol_det / 2 + det_pixel / 2)
detector_v0 = (-det_pixel * Nrow_det / 2 + det_pixel / 2)
u_coordinate = np.linspace(detector_u0, (detector_u0 + det_pixel * (Ncol_det - 1)), Ncol_det).astype(numpy.float32)
v_coordinate = np.linspace(detector_v0, (detector_v0 + det_pixel * (Nrow_det - 1)), Nrow_det).astype(numpy.float32)

ang_v=math.atan2(-detector_v0,D)
ang_u=math.atan2(-detector_u0,D)
print('ang_u:', np.rad2deg(ang_u))
print('ang_v:', np.rad2deg(ang_v))
print('50numpix_u:', 50*math.tan(ang_u))
print('50numpix_v:', 50*math.tan(ang_v))

proj = gpuarray.to_gpu(proj_cl)
tim_fl0 = time.time()

pi = np.float32(pi)
SincL_n = np.linspace(-N_angle, N_angle, (2 * N_angle + 1))
SincL = -2 / (pi * pi * abs(ds) * (4 * SincL_n * SincL_n - 1))
SincL = SincL.astype(numpy.float32)
print(SincL.max(),SincL.min())

tim_f_flgpu0 = time.time()
f_fl = gpuarray.to_gpu(np.empty((N_angle, Nrow_det, Ncol_det), np.float32))
tim_f_flgpu1 = time.time()

Ngridxy = Ncol_det / 512 * Nrow_det*N_angle
N_rank = math.floor(math.log2(Ngridxy)) + 1
N_rank_half = math.floor(N_rank / 2)
Gridy = 2 ** N_rank_half
Gridx = math.ceil(Ngridxy / Gridy)
print('fliter grid:', Gridx, Gridy)

fl(
    drv.In(N_angle), drv.In(Nrow_det), drv.In(Ncol_det), drv.In(SincL),
    proj, f_fl,
    grid=(Gridx, Gridy), block=(32, 16, 1))

del proj,proj_cl

# plt.figure()
# scale=int(N_angle/9)
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(f_fl.get()[i * scale, :, :], cmap="gray")
#     plt.title('f_fl ic=' + str(i * scale))
#
# plt.figure()
# plt.imshow(f_fl.get()[:, indr, :], cmap="gray")
# plt.title('f_fl ir=' + str(indr))

tim_fl1 = time.time()
print('convfl finish')
print('convfl time is :', tim_fl1 - tim_fl0)
print()

tim_bp0 = time.time()

Nz = np.int32(Nz)
Ny = np.int32(Ny)
Nx = np.int32(Nx)

pxx = np.linspace((-length / 2 + dx / 2), (length / 2 - dx / 2), Nx).astype(numpy.float32)
pyy = np.linspace((-width / 2 + dy / 2), (width / 2 - dy / 2), Ny).astype(numpy.float32)
pzz = np.linspace((-dz * Nz / 2 + dz / 2), (dz * Nz / 2 - dz / 2), Nz)+bis_z
pzz = pzz.astype(numpy.float32)

tim_recgpu0 = time.time()
rec = gpuarray.to_gpu(np.zeros((Nz, Ny, Nx), np.float32))
tim_recgpu1 = time.time()

Ngridxy = Ncol_det / 512 * Nz * Nx * Ny
N_rank = math.floor(math.log2(Ngridxy)) + 1
N_rank_half = math.floor(N_rank / 2)
Gridy = 2 ** N_rank_half
Gridx = math.ceil(Ngridxy / Gridy)
print('bp grid:', Gridx, Gridy)

bp(
    drv.In(pi),drv.In(R), drv.In(D), drv.In(Sx_init),
    drv.In(Nz), drv.In(Nx), drv.In(Ny),
    drv.In(N_angle), drv.In(Nrow_det),drv.In(Ncol_det),
    drv.In(phi), drv.In(u_coordinate), drv.In(v_coordinate),
    drv.In(pxx), drv.In(pyy),drv.In(pzz),
    f_fl,rec,
    grid=(Gridx, Gridy), block=(32, 16, 1))

tim_recget0 = time.time()
rec = rec.get()
tim_recget1 = time.time()
toc = time.time()
print('Get rec time is :', tim_recget1 - tim_recget0)
print('bp finish')
print('bp time is :', toc - tim_bp0)
print()
print('from weight to recon time is :', toc - tim_w0)
print('all time is :', toc - tic)

# rec=np.flip(np.transpose(rec,(0,2,1)),axis=2).copy()
print('rec max min:',rec.max(), rec.min())

if trans_state==1:
    rec = np.transpose(rec,(0,2,1)).copy()

plt.figure()
scale=int(Nz/9)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(rec[i * scale, :, :], cmap="gray")
    plt.title('recon z=' + str(i * scale))
    plt.axis('off')

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rec[5, :, :], cmap="gray")
plt.title('recon z=' + str(5*scale))
plt.subplot(1, 2, 2)
plt.imshow(rec[9, :, :], cmap="gray")
plt.title('recon z=' + str(8*scale))

plt.figure()
plt.imshow(rec[int(Nz/2), :, :], cmap="gray")
plt.title('z=' + str(int(Nz/2)))

indz=40
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(phantom[indz,:,:], cmap="gray")
plt.title('phantom iz=' + str(indz))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(rec[indz,:,:], cmap="gray")
plt.title('rec iz=' + str(indz))
plt.axis('off')

indy=187
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(phantom[:,indy,:], cmap="gray")
plt.title('phantom iy=' + str(indy))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(rec[:,indy,:], cmap="gray")
plt.title('rec iy=' + str(indy))
plt.axis('off')

indx=274
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(phantom[:,:,indx], cmap="gray")
plt.title('phantom ix=' + str(indx))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(rec[:,:,indx], cmap="gray")
plt.title('rec ix=' + str(indx))
plt.axis('off')


# zc = int(Nz / 2)+10
# xind = np.linspace(0, (Nx - 1), Nx)
# phantom_zc = phantom_data[zc, :, :]
# phantom_zc=np.flip(phantom_zc,axis=0)
# f_zc = rec[zc, :, :]
# delta_data = f_zc - phantom_zc
#
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(phantom_zc, cmap='gray')
# plt.title('phantom z=' + str(zc))
# # plt.colorbar(label='gray value')
# # plt.clim(0.8, 1.2)
# plt.subplot(2, 2, 3)
# plt.imshow(f_zc, cmap='gray')
# plt.title('recon z=' + str(zc))
# # plt.colorbar(label='gray value')
# # plt.clim(0.8,1.2)
# plt.subplot(2, 2, 4)
# plt.imshow(delta_data, cmap='gray')
# plt.title('delta')
# plt.subplot(2, 2, 2)
# ax = plt.subplot(2, 2, 2)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
# plt.rcParams['axes.unicode_minus'] = False
#
# # col=150  # alpha:透明度
# # ax.plot(xind, delta_data[0:Ny,col], '-', color='c', alpha=1, linewidth=1, label='delta')
# # ax.plot(xind, f_zc[0:Ny,col], '-', color='black', alpha=1, linewidth=1, label='DBP')
# # ax.plot(xind, phantom_zc[0:Ny,col], '-', color='red', alpha=1, linewidth=1, label='phantom')
# row = int(Ny/2)
# ax.plot(xind, delta_data[row, 0:Nx], '-', color='c', alpha=1, linewidth=1, label='delta')
# ax.plot(xind, f_zc[row, 0:Nx], '-', color='black', alpha=1, linewidth=1, label='DBP')
# ax.plot(xind, phantom_zc[row, 0:Nx], '-', color='red', alpha=1, linewidth=1, label='phantom')
# ax.legend(loc="best")
# plt.title('row='+str(row))
plt.show()

if ind_SaveRec==1:
    # image_recon = sitk.GetImageFromArray(rec)
    # sitk.WriteImage(image_recon, recon_path)
    # print('save:',recon_path)
    # del image_recon,recon_path
    out_dir = "./FBP_res/data"
    npy_out_path = os.path.join(out_dir, "vol_gt.npy")
    np.save(npy_out_path, rec)