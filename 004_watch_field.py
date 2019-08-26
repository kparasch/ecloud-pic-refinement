import TricubicInterpolation
import numpy as np
import matplotlib.pyplot as plt

import myfilemanager_sixtracklib as mfm
plt.style.use('kostas')


pinch = 'refined_pinch1_cut_mag4.0.h5'
pinches_folder = '../MyPinches/'

ob = mfm.h5_to_dict(pinches_folder+pinch)

xg = ob['xg']
yg = ob['yg']
zg = ob['zg']
phi = ob['phi']

print('ecloud:')
print(len(xg), len(yg), len(zg))
print('\t xg: %f'%xg[-1])
print('\t yg: %f'%yg[-1])
print('\t zg: %f'%zg[-1])
print('\t dx: %f'%(xg[1]-xg[0]))
print('\t dy: %f'%(yg[1]-yg[0]))
print('\t dz: %f'%(zg[1]-zg[0])) 

dx = xg[1] - xg[0]
dy = yg[1] - yg[0]
dz = zg[1] - zg[0]



TI = TricubicInterpolation.cTricubic.Tricubic_Interpolation(A=phi,
                                    x0 = xg[0], y0 = yg[0], z0 = zg[0],
                                    dx = xg[1] - xg[0],
                                    dy = yg[1] - yg[0],
                                    dz = zg[1] - zg[0],
                                    method = 'Exact')
TIdx = TricubicInterpolation.cTricubic.Tricubic_Interpolation(A=phi[:,:,:,1],
                                    x0 = xg[0], y0 = yg[0], z0 = zg[0],
                                    dx = xg[1] - xg[0],
                                    dy = yg[1] - yg[0],
                                    dz = zg[1] - zg[0],
                                    method = 'FD')

TIdy = TricubicInterpolation.cTricubic.Tricubic_Interpolation(A=phi[:,:,:,2],
                                    x0 = xg[0], y0 = yg[0], z0 = zg[0],
                                    dx = xg[1] - xg[0],
                                    dy = yg[1] - yg[0],
                                    dz = zg[1] - zg[0],
                                    method = 'FD')

#TIdz = TricubicInterpolation.cTricubic.Tricubic_Interpolation(A=phi[:,:,:,3],
TIdz = TricubicInterpolation.pyTricubic.Trilinear_Interpolation(A=phi[:,:,:,3],
                                    x0 = xg[0], y0 = yg[0], z0 = zg[0],
                                    dx = xg[1] - xg[0],
                                    dy = yg[1] - yg[0],
                                    dz = zg[1] - zg[0],
                                    method = 'FD')

xbox = 3.e-4
ybox = 1.e-3
zbox = 2.e-1

x_obs = dx/2.+10*dx
y_obs = dy/2.+10*dy
z_obs = dz/2.+dz/2.

xmask = abs(xg) < xbox
ymask = abs(yg) < ybox
zmask = abs(zg) < zbox
xo = xg[xmask]
yo = yg[ymask]
zo = zg[zmask]
xt = np.linspace(-xbox, xbox, 10000)
yt = np.linspace(-ybox, ybox, 10000)
zt = np.linspace(-zbox, zbox, 10000)
#####Ex#####
Ex_xt    = np.array([-TI.ddx(xx, y_obs, z_obs)   for xx in xt])
Ex_xo    = np.array([-TI.ddx(xx, y_obs, z_obs)   for xx in xo])
Ex_xt_ti = np.array([-TIdx.val(xx, y_obs, z_obs) for xx in xt])

Ex_yt    = np.array([-TI.ddx(x_obs, yy, z_obs)   for yy in yt])
Ex_yo    = np.array([-TI.ddx(x_obs, yy, z_obs)   for yy in yo])
Ex_yt_ti = np.array([-TIdx.val(x_obs, yy, z_obs) for yy in yt])

Ex_zt    = np.array([-TI.ddx(x_obs, y_obs, zz)   for zz in zt])
Ex_zo    = np.array([-TI.ddx(x_obs, y_obs, zz)   for zz in zo])
Ex_zt_ti = np.array([-TIdx.val(x_obs, y_obs, zz) for zz in zt])
#####Ey#####
Ey_xt    = np.array([-TI.ddy(xx, y_obs, z_obs)   for xx in xt])
Ey_xo    = np.array([-TI.ddy(xx, y_obs, z_obs)   for xx in xo])
Ey_xt_ti = np.array([-TIdy.val(xx, y_obs, z_obs) for xx in xt])

Ey_yt    = np.array([-TI.ddy(x_obs, yy, z_obs)   for yy in yt])
Ey_yo    = np.array([-TI.ddy(x_obs, yy, z_obs)   for yy in yo])
Ey_yt_ti = np.array([-TIdy.val(x_obs, yy, z_obs) for yy in yt])

Ey_zt    = np.array([-TI.ddy(x_obs, y_obs, zz)   for zz in zt])
Ey_zo    = np.array([-TI.ddy(x_obs, y_obs, zz)   for zz in zo])
Ey_zt_ti = np.array([-TIdy.val(x_obs, y_obs, zz) for zz in zt])
#####Ez#####
Ez_xt    = np.array([-TI.ddz(xx, y_obs, z_obs)   for xx in xt])
Ez_xo    = np.array([-TI.ddz(xx, y_obs, z_obs)   for xx in xo])
Ez_xt_ti = np.array([-TIdz.val(xx, y_obs, z_obs) for xx in xt])

Ez_yt    = np.array([-TI.ddz(x_obs, yy, z_obs)   for yy in yt])
Ez_yo    = np.array([-TI.ddz(x_obs, yy, z_obs)   for yy in yo])
Ez_yt_ti = np.array([-TIdz.val(x_obs, yy, z_obs) for yy in yt])

Ez_zt    = np.array([-TI.ddz(x_obs, y_obs, zz)   for zz in zt])
Ez_zo    = np.array([-TI.ddz(x_obs, y_obs, zz)   for zz in zo])
Ez_zt_ti = np.array([-TIdz.val(x_obs, y_obs, zz) for zz in zt])

maxEx = 1.1*max([max(abs(Ex_xt)),max(abs(Ex_yt)),max(abs(Ex_zt))])
fig = plt.figure(1,figsize=[18,5])
ax1 = fig.add_subplot(131)
ax1.plot(xt, Ex_xt,'r')
ax1.plot(xo, Ex_xo,'b.')
ax1.plot(xt, Ex_xt_ti,'k--')
ax1.set_xlabel('x')
ax1.set_ylabel('Ex')
ax1.set_ylim(-maxEx, maxEx)

ax2 = fig.add_subplot(132)
ax2.plot(yt, Ex_yt,'r')
ax2.plot(yo, Ex_yo,'b.')
ax2.plot(yt, Ex_yt_ti,'k--')
ax2.set_xlabel('y')
ax2.set_ylim(-maxEx, maxEx)

ax3 = fig.add_subplot(133)
ax3.plot(zt, Ex_zt,'r')
ax3.plot(zo, Ex_zo,'b.')
ax3.plot(zt, Ex_zt_ti,'k--')
ax3.set_xlabel('z')
ax3.set_ylim(-maxEx, maxEx)


maxEy = 1.1*max([max(abs(Ey_xt)),max(abs(Ey_yt)),max(abs(Ey_zt))])
fig2 = plt.figure(2,figsize=[18,5])
ax21 = fig2.add_subplot(131)
ax21.plot(xt, Ey_xt,'r')
ax21.plot(xo, Ey_xo,'b.')
ax21.plot(xt, Ey_xt_ti,'k--')
ax21.set_xlabel('x')
ax21.set_ylabel('Ey')
ax21.set_ylim(-maxEy, maxEy)

ax22 = fig2.add_subplot(132)
ax22.plot(yt, Ey_yt,'r')
ax22.plot(yo, Ey_yo,'b.')
ax22.plot(yt, Ey_yt_ti,'k--')
ax22.set_xlabel('y')
ax22.set_ylim(-maxEy, maxEy)

ax23 = fig2.add_subplot(133)
ax23.plot(zt, Ey_zt,'r')
ax23.plot(zo, Ey_zo,'b.')
ax23.plot(zt, Ey_zt_ti,'k--')
ax23.set_xlabel('z')
ax23.set_ylim(-maxEy, maxEy)


maxEz = 1.1*max([max(abs(Ez_xt)),max(abs(Ez_yt)),max(abs(Ez_zt))])
fig3 = plt.figure(3,figsize=[18,5])
ax31 = fig3.add_subplot(131)
ax31.plot(xt, Ez_xt,'r')
ax31.plot(xo, Ez_xo,'b.')
ax31.plot(xt, Ez_xt_ti,'k--')
ax31.set_xlabel('x')
ax31.set_ylabel('Ez')
ax31.set_ylim(-maxEz, maxEz)

ax32 = fig3.add_subplot(132)
ax32.plot(yt, Ez_yt,'r')
ax32.plot(yo, Ez_yo,'b.')
ax32.plot(yt, Ez_yt_ti,'k--')
ax32.set_xlabel('y')
ax32.set_ylim(-maxEz, maxEz)

ax33 = fig3.add_subplot(133)
ax33.plot(zt, Ez_zt,'r')
ax33.plot(zo, Ez_zo,'b.')
ax33.plot(zt, Ez_zt_ti,'k--')
ax33.set_xlabel('z')
ax33.set_ylim(-maxEz, maxEz)
plt.show()
