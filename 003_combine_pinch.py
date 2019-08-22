import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

import myfilemanager as mfm
import myfilemanager_sixtracklib as mfm_stl
import mystyle as ms
from TricubicInterpolation import cTricubic as ti

import PyPIC.PyPIC_Scatter_Gather as PyPICSC
import PyPIC.geom_impact_poly as poly
from PyPIC.MultiGrid import AddInternalGrid


def setup_pic(fname, magnify=2., N_nodes_discard=10):

    ob = mfm.myloadmat_to_obj(fname)
    Dh_magnify = (ob.xg[1]-ob.xg[0])/magnify
    x_magnify = -ob.xg[N_nodes_discard]
    y_magnify = -ob.yg[N_nodes_discard]
    
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

def get_slice(picoutside, picinside, fname, islice):
    
    ob = mfm.myloadmat_to_obj(fname)
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

#ob = mfm.myloadmat_to_obj('pinch_cut.mat')

if not os.path.exists('temp'):
    os.mkdir('temp')
fname = 'pinch_pic_data_mean2.mat'
N_nodes_discard = 10
magnify = 2.
pic_out, pic_in, zg = setup_pic(fname, magnify=magnify, N_nodes_discard=N_nodes_discard)

print('Max x = %f'%abs(pic_out.xg[0]))
print('Max y = %f'%abs(pic_out.yg[0]))
dx_in = pic_in.xg[1] - pic_in.xg[0]
dy_in = pic_in.yg[1] - pic_in.yg[0]
dz = zg[1] - zg[0]

x0_in = pic_in.xg[0]
y0_in = pic_in.xg[0]

slices = np.zeros([4,pic_in.Nxg, pic_in.Nyg])
Nd = N_nodes_discard + 1
phi_slice_exact = np.zeros([pic_out.Nxg-2*Nd, pic_out.Nyg-2*Nd, 8])
xg_e = pic_out.xg[Nd:-Nd]
yg_e = pic_out.yg[Nd:-Nd]
zg_e = zg[2:-2]
n_slices = len(zg)

print('Number of slices: {}'.format(n_slices))

nx = pic_out.Nxg-2*Nd
ny = pic_out.Nyg-2*Nd
#x0 = pic.out.xg[Nd]
#y0 = pic.out.yg[Nd]
#z0 = pic.out.zg[1:-2]
del pic_out
del pic_in
del slices

phi_e = np.zeros([n_slices, nx, ny, 8])
for i in range(1,n_slices-3):
    print('Reading Slice: %d'%i)
    ob = mfm.myloadmat_to_obj('temp/slice%d.mat'%i)
    phi_e[i-1,:,:,:] = ob.phi_slice_exact[:,:,:]
    del ob

dd = {'xg' : xg_e,
      'yg' : yg_e,
      'zg' : zg_e,
      'phi' : phi_e
     }

print('Begin saving..')
compression_opts = 0
mfm_stl.dict_to_h5(dd, 'refined_exact_pinch.h5', compression_opts=compression_opts)
#sio.savemat('refined_exact_pinch.mat',dd, oned_as = 'row')
