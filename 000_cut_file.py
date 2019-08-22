import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import myfilemanager as mfm

ob = mfm.myloadmat_to_obj('pinch_pic_data_mean2.mat')

i_zero = np.argmin(np.abs(ob.xg))
j_zero = np.argmin(np.abs(ob.yg))

N_keep_h = 100
N_keep_v = 90

dict_new_file = {}
dict_new_file['Ex' ] = ob.Ex[:, i_zero-N_keep_h:i_zero+N_keep_h+1, j_zero-N_keep_v:j_zero+N_keep_v+1] 
dict_new_file['Ey' ] = ob.Ey[:, i_zero-N_keep_h:i_zero+N_keep_h+1, j_zero-N_keep_v:j_zero+N_keep_v+1] 
dict_new_file['phi'] = ob.phi[:, i_zero-N_keep_h:i_zero+N_keep_h+1, j_zero-N_keep_v:j_zero+N_keep_v+1] 
dict_new_file['rho'] = ob.rho[:, i_zero-N_keep_h:i_zero+N_keep_h+1, j_zero-N_keep_v:j_zero+N_keep_v+1] 
dict_new_file['xg' ] = ob.xg[i_zero-N_keep_h:i_zero+N_keep_h+1] 
dict_new_file['yg' ] = ob.yg[j_zero-N_keep_v:j_zero+N_keep_v+1]
dict_new_file['zg' ] = ob.zg

sio.savemat('pinch_cut.mat', dict_new_file, oned_as='row')

