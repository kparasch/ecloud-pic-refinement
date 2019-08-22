import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import myfilemanager as mfm
import mystyle as ms
from TricubicInterpolation import tricubic_interpolation as ti

import PyPIC.PyPIC_Scatter_Gather as PyPICSC
import PyPIC.geom_impact_poly as poly
from PyPIC.MultiGrid import AddInternalGrid

ob = mfm.myloadmat_to_obj('pinch_cut.mat')

interp2d = True 


i_slice = 250
z_obs = ob.zg[i_slice]

x_magnify = 5e-4
y_magnify = 4.5e-4
Dh_magnify = 5e-6


if interp2d:
    for kk, zz in enumerate(ob.zg):
        ob.rho[kk, :, :] = ob.rho[i_slice, :, :]
        ob.phi[kk, :, :] = ob.phi[i_slice, :, :]
        ob.Ex[kk, :, :] = ob.Ex[i_slice, :, :]
        ob.Ey[kk, :, :] = ob.Ey[i_slice, :, :]


# Interpolation on initial grid
tinterp = ti.Tricubic_Interpolation(A=ob.phi.transpose(1,2,0)[:, :, i_slice-5:i_slice+6], 
        x0=ob.xg[0], y0=ob.yg[0], z0=ob.zg[i_slice-5],
        dx=ob.xg[1]-ob.xg[0], dy=ob.yg[1]-ob.yg[0], dz=ob.zg[1]-ob.zg[0])

x_tint = np.linspace(-4.5e-4, 4.5e-4, 1000)
y_tint = 0.*x_tint
z_tint = 0.*x_tint + z_obs

phi_tint = 0.*x_tint
Ex_tint = 0.*x_tint

for ii, (xx, yy, zz) in enumerate(zip(x_tint, y_tint, z_tint)):
    phi_tint[ii] = tinterp.val(xx, yy, zz)
    Ex_tint[ii] = -tinterp.ddx(xx, yy, zz)

# 2D PIC
assert((ob.xg[1] - ob.xg[0]) == (ob.yg[1] - ob.yg[0]))
na = np.array
chamb = poly.polyg_cham_geom_object({'Vx':na([ob.xg[-1], ob.xg[0], ob.xg[0], ob.xg[-1]]),
                                       'Vy':na([ob.yg[-1], ob.yg[-1], ob.yg[0],ob.yg[0]]),
                                       'x_sem_ellip_insc':1e-3,
                                       'y_sem_ellip_insc':1e-3})

pic = PyPICSC.PyPIC_Scatter_Gather(xg=ob.xg, yg=ob.yg)
pic.phi = ob.phi[i_slice, :, :]
pic.efx = ob.Ex[i_slice, :, :]
pic.efy = ob.Ey[i_slice, :, :]
pic.rho = ob.rho[i_slice, :, :]
pic.chamb = chamb

Ex_picint, _ = pic.gather(x_tint, y_tint)


# Internal pic
picdg = AddInternalGrid(pic, 
        x_min_internal=-x_magnify,
        x_max_internal=x_magnify, 
        y_min_internal=-y_magnify, 
        y_max_internal=y_magnify, 
        Dh_internal=Dh_magnify, 
        N_nodes_discard = 10)
picinside = picdg.pic_internal 

picinside.rho = np.reshape(pic.gather_rho(picinside.xn, picinside.yn),
        (picinside.Nxg, picinside.Nyg))
picinside.solve(flag_verbose = True, pic_external=pic)

rho_insideint = picinside.gather_rho(x_tint, y_tint)
Ex_inside, _= picinside.gather(x_tint, y_tint)
phi_inside = picinside.gather_phi(x_tint, y_tint)

# Tricubic on internal pic
tinterp_inside = ti.Tricubic_Interpolation(A=na(tinterp.A.shape[2]*[picinside.phi]).transpose(1,2,0), 
        x0=picinside.xg[0], y0=picinside.yg[0], z0=-tinterp.A.shape[2]/2.,
        dx=picinside.xg[1]-picinside.xg[0], dy=picinside.yg[1]-picinside.yg[0], dz=1.)

phi_tinside = 0.*x_tint
Ex_tinside = 0.*x_tint

for ii, (xx, yy, zz) in enumerate(zip(x_tint, y_tint, z_tint)):
    phi_tinside[ii] = tinterp_inside.val(xx, yy, zz)
    Ex_tinside[ii] = -tinterp_inside.ddx(xx, yy, zz)

# Tricubic with new derivatives
Nx,Ny,Nz = ob.phi.transpose(1,2,0)[:, :, i_slice-5:i_slice+6].shape
A = np.zeros([Nx,Ny,Nz,8])

A[:,:,:,0] = ob.phi.transpose(1,2,0)[:, :, i_slice-5:i_slice+6]

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            if tinterp_inside.is_inside_box(ob.xg[ii], ob.yg[jj], z_obs):
                A[ii,jj,kk,0] = tinterp_inside.val(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,1] = tinterp_inside.ddx(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,2] = tinterp_inside.ddy(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,3] = tinterp_inside.ddz(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,4] = tinterp_inside.ddxdy(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,5] = tinterp_inside.ddxdz(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,6] = tinterp_inside.ddydz(ob.xg[ii], ob.yg[jj], z_obs)
                A[ii,jj,kk,7] = tinterp_inside.ddxdydz(ob.xg[ii], ob.yg[jj], z_obs)
            else:
                A[ii,jj,kk,:] = 0.

#DD = A[:,:,:,0]
tinterp_der = ti.Tricubic_Interpolation(A=A, 
        x0=ob.xg[0], y0=ob.yg[0], z0=ob.zg[i_slice-5],
        dx=ob.xg[1]-ob.xg[0], dy=ob.yg[1]-ob.yg[0], dz=ob.zg[1]-ob.zg[0], method='Exact')
#tinterp_der = ti.Tricubic_Interpolation(A=DD, 
#        x0=ob.xg[0], y0=ob.yg[0], z0=ob.zg[i_slice-5],
#        dx=ob.xg[1]-ob.xg[0], dy=ob.yg[1]-ob.yg[0], dz=ob.zg[1]-ob.zg[0])




phi_tder = 0.*x_tint
Ex_tder = 0.*x_tint

for ii, (xx, yy, zz) in enumerate(zip(x_tint, y_tint, z_tint)):
    phi_tder[ii] = tinterp_der.val(xx, yy, zz)
    Ex_tder[ii] = -tinterp_der.ddx(xx, yy, zz)



# Plotting
plt.close('all')
#ms.mystyle_arial()
plt.style.use('kostas')
ax1 =  None
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(1,1,1)
# ax1.pcolormesh(ob.xg, ob.yg, ob.rho[i_slice, :, :].T)
# 
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(1,1,1, sharex=ax1)
# ax2.pcolormesh(ob.xg, ob.yg, ob.Ex[i_slice, :, :].T)

y_obs = 0.
j_obs = np.argmin(np.abs(ob.yg - y_obs))

fig3 = plt.figure(3)
fig3.set_facecolor('w')
ax31 = fig3.add_subplot(3,1,1, sharex=ax1)
ax32 = fig3.add_subplot(3,1,2, sharex=ax31)
ax33 = fig3.add_subplot(3,1,3, sharex=ax31)

ax31.plot(ob.xg, ob.rho[i_slice, :, j_obs], '.')
ax31.plot(x_tint, rho_insideint, 'kx--')

ax32.plot(ob.xg, ob.phi[i_slice, :, j_obs],'.-')
ax32.plot(x_tint, phi_inside, 'k')
ax32.plot(x_tint, phi_tinside, 'orange')
ax32.plot(x_tint, phi_tder, 'm')
ax33.plot(ob.xg, ob.Ex[i_slice, :, j_obs], '.')

ax32.plot(x_tint, phi_tint,'r-')

ax33.plot(x_tint, Ex_tint, 'r-')
ax33.plot(x_tint, Ex_picint, 'g--')
ax33.plot(x_tint, Ex_inside, 'k')
ax33.plot(x_tint, Ex_tinside, 'orange')
ax33.plot(x_tint, Ex_tder, 'm')

plt.show()
