import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time
import shutil

import myfilemanager as mfm
import myfilemanager_sixtracklib as mfm_stl
from TricubicInterpolation import cTricubic as ti

import PyPIC.PyPIC_Scatter_Gather as PyPICSC
import PyPIC.geom_impact_poly as poly
from PyPIC.MultiGrid import AddInternalGrid

def symmetrize(A):
    shape = A.shape
    assert len(shape) == 2
    assert shape[0] % 2 == 1
    assert shape[1] % 2 == 1
    ix0 = shape[0] // 2
    iy0 = shape[1] // 2

    newA = np.empty([ix0+1,iy0+1])
    newA[0,0] = A[ix0,iy0]
    newA[1:,0] = 0.5*(A[0:ix0,iy0][::-1] + A[ix0+1:,iy0])
    newA[0,1:] = 0.5*(A[ix0,0:iy0][::-1] + A[ix0,iy0+1:])
    newA[1:,1:] = 0.25*( A[ix0+1:,iy0+1:] +
                         A[0:ix0,iy0+1:][::-1,:] + 
                         A[ix0+1:,0:iy0][:,::-1] + 
                         A[0:ix0,0:iy0][::-1,::-1]
                       )

    retA=np.empty_like(A)

    retA[ix0,iy0] = newA[0,0]

    retA[ix0+1:,iy0] = newA[1:,0]
    retA[0:ix0,iy0] = newA[1:,0][::-1]
    retA[ix0,iy0+1:] = newA[0,1:]
    retA[ix0,0:iy0] = newA[0,1:][::-1]

    retA[ix0+1:,iy0+1:] = newA[1:,1:]
    retA[0:ix0,iy0+1:] = newA[1:,1:][::-1,:]
    retA[ix0+1:,0:iy0] = newA[1:,1:][:,::-1]
    retA[0:ix0,0:iy0] = newA[1:,1:][::-1,::-1]

    return retA

def setup_pic(fname, magnify=2., N_nodes_discard=10, symmetric_slice_2D=True):

    ob = mfm.myloadmat_to_obj(fname)
    Dh_magnify = (ob.xg[1]-ob.xg[0])/magnify
    x_magnify = -ob.xg[N_nodes_discard]
    y_magnify = -ob.yg[N_nodes_discard]
    
    if symmetric_slice_2D:
    	pic_rho = symmetrize(ob.rho[0,:,:])
    	pic_phi = symmetrize(ob.phi[0,:,:])
    else:
    	pic_rho = ob.rho[0,:,:].copy()
    	pic_phi = ob.phi[0,:,:].copy()
    xg_out = ob.xg.copy()
    yg_out = ob.yg.copy()
    zg_out = ob.zg.copy()
    del ob

    chamb = poly.polyg_cham_geom_object({'Vx':np.array([xg_out[-1], xg_out[0], xg_out[0], xg_out[-1]]),
                                       'Vy':np.array([yg_out[-1], yg_out[-1], yg_out[0], yg_out[0]]),
                                       'x_sem_ellip_insc':1e-3,
                                       'y_sem_ellip_insc':1e-3})

    pic = PyPICSC.PyPIC_Scatter_Gather(xg=xg_out, yg = yg_out)
    pic.phi = pic_phi
    #pic.efx = ob.Ex[0, :, :]
    #pic.efy = ob.Ey[0, :, :]
    pic.rho = pic_rho
    pic.chamb = chamb
    
    #Ex_picint, _ = pic.gather(x_tint, y_tint)
    
    
    # Internal pic
    picdg = AddInternalGrid(pic, 
            x_min_internal=-x_magnify,
            x_max_internal=x_magnify, 
            y_min_internal=-y_magnify, 
            y_max_internal=y_magnify, 
            Dh_internal=Dh_magnify, 
            N_nodes_discard = N_nodes_discard)
    picinside = picdg.pic_internal 
    
    picinside.rho = np.reshape(pic.gather_rho(picinside.xn, picinside.yn),
            (picinside.Nxg, picinside.Nyg))
    picinside.solve(flag_verbose = True, pic_external=pic)
    
    #rho_insideint = picinside.gather_rho(x_tint, y_tint)
    #Ex_inside, _ = picinside.gather(x_tint, y_tint)
    #phi_inside = picinside.gather_phi(x_tint, y_tint)

    return pic, picinside, zg_out

def get_slice(picoutside, picinside, fname, islice, symmetric_slice_2D=True):
    
    ob = mfm.myloadmat_to_obj(fname)
    if symmetric_slice_2D:
    	rho = symmetrize(ob.rho[islice, :, :])
    	phi = symmetrize(ob.phi[islice, :, :])
    else:
    	rho = ob.rho[islice, :, :].copy()
    	phi = ob.phi[islice, :, :].copy()
    del ob

    picoutside.phi = phi
    picoutside.rho = rho

    picinside.rho = np.reshape( picoutside.gather_rho(picinside.xn, picinside.yn),
                               (picinside.Nxg, picinside.Nyg))
    picinside.solve(flag_verbose = True, pic_external = picoutside)

    phi_refined = picinside.gather_phi(picinside.xn, picinside.yn)

    return phi_refined.reshape(picinside.Nxg, picinside.Nyg)

def phi_n_e_slices(pic_out, pic_in, fname, islice, dz, max_slice, symmetric_slice_2D=True):
    slices = np.zeros([3,pic_in.Nxg, pic_in.Nyg])
    slices[0,:,:] = get_slice(pic_out, pic_in, fname, islice - 1, symmetric_slice_2D) 
    slices[1,:,:] = get_slice(pic_out, pic_in, fname, islice, symmetric_slice_2D) 
    if islice + 1 < max_slice:
        slices[2,:,:] = get_slice(pic_out, pic_in, fname, islice + 1, symmetric_slice_2D) 

    ex_slice = np.zeros([pic_in.Nxg,pic_in.Nyg])
    ey_slice = np.zeros([pic_in.Nxg,pic_in.Nyg])
    ez_slice = np.zeros([pic_in.Nxg,pic_in.Nyg])

    ex_slice[1:-1,:] = (0.5*(slices[1,2:,:] - slices[1,0:-2,:]))/pic_in.Dh
    ey_slice[:,1:-1] = (0.5*(slices[1,:,2:] - slices[1,:,0:-2]))/pic_in.Dh 
    ez_slice[:,:] = (0.5*(slices[2,:,:] - slices[0,:,:]))/dz
    return slices[1,:,:], ex_slice, ey_slice, ez_slice

#ob = mfm.myloadmat_to_obj('pinch_cut.mat')
start_time = time.time()
N_nodes_discard = 10
magnify = 4
do_kicks = True
symmetric2D = True
#symmetric2D=True

symm_str = ''
if symmetric2D: symm_str = '_symm2D'

pinch_in = 'pinch1_cut'
fname = 'eclouds/' + pinch_in + '.mat'
pinch_out = 'refined_'+pinch_in+symm_str+'_mag%.1f.h5'%magnify
temp_folder = pinch_out+'.temp'
if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)

print('Pinch in: ' + pinch_in)
print('Magnify = %f'%magnify)
compression_opts = 0
pic_out, pic_in, zg = setup_pic(fname, magnify=magnify, N_nodes_discard=N_nodes_discard, symmetric_slice_2D=symmetric2D)

print('Max x = %f'%abs(pic_out.xg[0]))
print('Max y = %f'%abs(pic_out.yg[0]))
dx_in = pic_in.xg[1] - pic_in.xg[0]
dy_in = pic_in.yg[1] - pic_in.yg[0]
dz = zg[1] - zg[0]

x0_in = pic_in.xg[0]
y0_in = pic_in.yg[0]

slices = np.zeros([4,pic_in.Nxg, pic_in.Nyg])
if do_kicks:
    ex_slices = np.zeros([4,pic_in.Nxg, pic_in.Nyg])
    ey_slices = np.zeros([4,pic_in.Nxg, pic_in.Nyg])
    ez_slices = np.zeros([4,pic_in.Nxg, pic_in.Nyg])
Nd = N_nodes_discard + 1
phi_slice_exact = np.zeros([pic_out.Nxg-2*Nd, pic_out.Nyg-2*Nd, 8])
if do_kicks:
    ex_slice_exact = np.zeros([pic_out.Nxg-2*Nd, pic_out.Nyg-2*Nd, 8])
    ey_slice_exact = np.zeros([pic_out.Nxg-2*Nd, pic_out.Nyg-2*Nd, 8])
    ez_slice_exact = np.zeros([pic_out.Nxg-2*Nd, pic_out.Nyg-2*Nd, 8])
xg_e = pic_out.xg[Nd:-Nd]
yg_e = pic_out.yg[Nd:-Nd]
zg_e = zg[2:-2]
n_slices = len(zg)

print('Number of slices: {}'.format(n_slices))

for islice in [1,2,3]:
    if do_kicks:
        slices[islice-1, :, :], ex_slices[islice-1, :, :], ey_slices[islice-1, :, :], ez_slices[islice-1, :, :] = phi_n_e_slices(pic_out, pic_in, fname, islice, dz, n_slices, symmetric_slice_2D=symmetric2D)
    else:
        slices[islice-1,:,:] = get_slice(pic_out, pic_in, fname, islice, symmetric_slice_2D=symmetric2D)

for k in range(1,n_slices-3):
    z0_in = zg[k] 
    print('Slice {}/{}'.format(k-1, n_slices-4))
    #slices[3] = get_slice(pic_out, pic_in, fname, k+3)
    if do_kicks:
        slices[3, :, :], ex_slices[3, :, :], ey_slices[3, :, :], ez_slices[3, :, :] = phi_n_e_slices(pic_out, pic_in, fname, k+3, dz, n_slices, symmetric_slice_2D=symmetric2D)
    else:
        slices[3,:,:] = get_slice(pic_out, pic_in, fname, k+3, symmetric_slice_2D=symmetric2D)
    tinterp = ti.Tricubic_Interpolation(A=slices, 
            x0=z0_in, y0=x0_in, z0=y0_in,
            dx=dz, dy=dx_in, dz=dy_in)
    if do_kicks:
        tinterp_dx = ti.Tricubic_Interpolation(A=ex_slices, 
                x0=z0_in, y0=x0_in, z0=y0_in,
                dx=dz, dy=dx_in, dz=dy_in)
        tinterp_dy = ti.Tricubic_Interpolation(A=ey_slices, 
                x0=z0_in, y0=x0_in, z0=y0_in,
                dx=dz, dy=dx_in, dz=dy_in)
        tinterp_dz = ti.Tricubic_Interpolation(A=ez_slices, 
                x0=z0_in, y0=x0_in, z0=y0_in,
                dx=dz, dy=dx_in, dz=dy_in)

    for i in range(Nd, pic_out.Nxg - Nd):
        ii = i - Nd
        for j in range(Nd, pic_out.Nyg - Nd):
            jj = j - Nd
#            is_inside = tinterp.is_inside_box( z0_in + dz, pic_out.xg[i], pic_out.yg[j])
#            if not is_inside: 
#                ix, iy, iz = tinterp.coords_to_indices( z0_in + dz, pic_out.xg[i], pic_out.yg[j])
#                print(ix, iy, iz)
#                continue
            phi_slice_exact[ii,jj,0] = tinterp.val(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,1] = tinterp.ddy(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,2] = tinterp.ddz(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,3] = tinterp.ddx(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,4] = tinterp.ddydz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,5] = tinterp.ddxdy(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,6] = tinterp.ddxdz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            phi_slice_exact[ii,jj,7] = tinterp.ddxdydz( z0_in + dz, pic_out.xg[i], pic_out.yg[j])
            if do_kicks:
                ex_slice_exact[ii,jj,0] = tinterp_dx.val(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,1] = tinterp_dx.ddy(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,2] = tinterp_dx.ddz(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,3] = tinterp_dx.ddx(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,4] = tinterp_dx.ddydz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,5] = tinterp_dx.ddxdy(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,6] = tinterp_dx.ddxdz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ex_slice_exact[ii,jj,7] = tinterp_dx.ddxdydz( z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,0] = tinterp_dy.val(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,1] = tinterp_dy.ddy(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,2] = tinterp_dy.ddz(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,3] = tinterp_dy.ddx(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,4] = tinterp_dy.ddydz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,5] = tinterp_dy.ddxdy(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,6] = tinterp_dy.ddxdz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ey_slice_exact[ii,jj,7] = tinterp_dy.ddxdydz( z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,0] = tinterp_dz.val(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,1] = tinterp_dz.ddy(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,2] = tinterp_dz.ddz(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,3] = tinterp_dz.ddx(     z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,4] = tinterp_dz.ddydz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,5] = tinterp_dz.ddxdy(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,6] = tinterp_dz.ddxdz(   z0_in + dz, pic_out.xg[i], pic_out.yg[j])
                ez_slice_exact[ii,jj,7] = tinterp_dz.ddxdydz( z0_in + dz, pic_out.xg[i], pic_out.yg[j])


    #sio.savemat('temp/slice%d.mat'%k, {'phi_slice_exact' : phi_slice_exact}, oned_as = 'row')
    mfm_stl.dict_to_h5({'phi_slice_exact' : phi_slice_exact}, temp_folder+'/slice%d.h5'%k, compression_opts=compression_opts)
    if do_kicks:
        mfm_stl.dict_to_h5({'ex_slice_exact' : ex_slice_exact}, temp_folder+'/ex_slice%d.h5'%k, compression_opts=compression_opts)
        mfm_stl.dict_to_h5({'ey_slice_exact' : ey_slice_exact}, temp_folder+'/ey_slice%d.h5'%k, compression_opts=compression_opts)
        mfm_stl.dict_to_h5({'ez_slice_exact' : ez_slice_exact}, temp_folder+'/ez_slice%d.h5'%k, compression_opts=compression_opts)

    slices[0, :, :] = slices[1, :, :]
    slices[1, :, :] = slices[2, :, :]
    slices[2, :, :] = slices[3, :, :]
    if do_kicks:
        ex_slices[0, :, :] = ex_slices[1, :, :]
        ex_slices[1, :, :] = ex_slices[2, :, :]
        ex_slices[2, :, :] = ex_slices[3, :, :]
        ey_slices[0, :, :] = ey_slices[1, :, :]
        ey_slices[1, :, :] = ey_slices[2, :, :]
        ey_slices[2, :, :] = ey_slices[3, :, :]
        ez_slices[0, :, :] = ez_slices[1, :, :]
        ez_slices[1, :, :] = ez_slices[2, :, :]
        ez_slices[2, :, :] = ez_slices[3, :, :]
nx = pic_out.Nxg-2*Nd
ny = pic_out.Nyg-2*Nd
#x0 = pic.out.xg[Nd]
#y0 = pic.out.yg[Nd]
#z0 = pic.out.zg[1:-2]
del pic_out
del pic_in
del slices
if do_kicks:
    del ex_slices
    del ey_slices
    del ez_slices

phi_e = np.zeros([n_slices-4, nx, ny, 8])
if do_kicks:
    ex_e = np.zeros([n_slices-4, nx, ny, 8])
    ey_e = np.zeros([n_slices-4, nx, ny, 8])
    ez_e = np.zeros([n_slices-4, nx, ny, 8])
for i in range(1,n_slices-3):
    print('Reading Slice: %d'%i)
    ob = mfm_stl.h5_to_dict(temp_folder+'/slice%d.h5'%i)
    phi_e[i-1,:,:,:] = ob['phi_slice_exact'][:,:,:]
    #ob = mfm.myloadmat_to_obj('temp/slice%d.mat'%i)
    #phi_e[i-1,:,:,:] = ob.phi_slice_exact[:,:,:]
    del ob
    if do_kicks:
        ob_ex = mfm_stl.h5_to_dict(temp_folder+'/ex_slice%d.h5'%i)
        ex_e[i-1,:,:,:] = ob_ex['ex_slice_exact'][:,:,:]
        del ob_ex
        ob_ey = mfm_stl.h5_to_dict(temp_folder+'/ey_slice%d.h5'%i)
        ey_e[i-1,:,:,:] = ob_ey['ey_slice_exact'][:,:,:]
        del ob_ey
        ob_ez = mfm_stl.h5_to_dict(temp_folder+'/ez_slice%d.h5'%i)
        ez_e[i-1,:,:,:] = ob_ez['ez_slice_exact'][:,:,:]
        del ob_ez
        
if symmetric2D:
    xg_e = xg_e[nx//2:]
    yg_e = yg_e[ny//2:]
    phi_e = phi_e[:,nx//2:,ny//2:,:]

    phi_e[:,0,:,1] = 0.
    phi_e[:,:,0,2] = 0.

    phi_e[:,0,:,4] = 0.
    phi_e[:,:,0,4] = 0.

    phi_e[:,0,:,5] = 0.

    phi_e[:,:,0,6] = 0.

    phi_e[:,0,:,7] = 0.
    phi_e[:,:,0,7] = 0.

    if do_kicks:
    	ex_e = ex_e[:,nx//2:,ny//2:,:]
    	ey_e = ey_e[:,nx//2:,ny//2:,:]
    	ez_e = ez_e[:,nx//2:,ny//2:,:]

dd = {'xg' : xg_e,
      'yg' : yg_e,
      'zg' : zg_e,
      'phi' : phi_e.transpose(1,2,0,3)
     }

if do_kicks:
    dd_e = {'xg' : xg_e,
             'yg' : yg_e,
             'zg' : zg_e,
             'ex' : ex_e.transpose(1,2,0,3),
             'ey' : ey_e.transpose(1,2,0,3),
             'ez' : ez_e.transpose(1,2,0,3)
            }

print('Begin saving..')
mfm_stl.dict_to_h5(dd, 'eclouds/'+pinch_out, compression_opts=compression_opts)
if do_kicks:
    mfm_stl.dict_to_h5(dd_e, 'eclouds/e_'+pinch_out, compression_opts=compression_opts)
end_time = time.time()
shutil.rmtree(temp_folder)
print('Running time: %f mins'%((end_time-start_time)/60.))

