import glob
import os
import numpy as np

if not os.path.isdir('Data'):
    os.mkdir('Data')

LCDM = False
if LCDM:
    tag = 'StandardUniverse_CL_Table'
    filesLoad = glob.glob('../OutputFiles/' + tag + '*.dat')
    fileOut = 'Data/LCDM_TT_Cls.dat'
else:
    tag = 'MultiBrane_CL_Table'
    filesLoad = glob.glob('../OutputFiles/' + tag + '*.dat')
    f_ex = filesLoad[0]
    nbr = float(f_ex[f_ex.find('_Nbrane_')+8:f_ex.find('_Press')])
    nexCDM = float(f_ex[f_ex.find('_eCDM_')+6:f_ex.find('.dat')])
    fileOut = 'Data/MultiB_Nbr_{:.1e}_Ecdm_{:.3e}_TT_Cls.dat'.format(nbr, nexCDM)

combine_files = False

if combine_files:
    existingF = np.loadtxt(fileOut)

data_collect = []

for f in filesLoad:
    dataL = np.loadtxt(f)
    omegaB = float(f[f.find('_Ob_')+4:f.find('_Oc_')])
    omegaC = float(f[f.find('_Oc_')+4:f.find('_H0_')])
    hub = float(f[f.find('_H0_')+4:f.find('_Neff_')])
    ns = float(f[f.find('_Ns_')+4:f.find('_As_')])
    As = float(f[f.find('_As_')+4:f.find('_zreion_')])
    zre = float(f[f.find('_zreion_')+8:f.find('__Nbrane')])
    if not LCDM:
        pressF = float(f[f.find('PressFac_')+9:f.find('_eCDM_')])
    
    ell_vals = dataL[:, 0]
    cls = np.log10(dataL[:,1] * 1e12)
    if LCDM:
        vals = [omegaB, omegaC, hub, ns, np.log10(As), zre]
    else:
        vals = [np.log10(pressF), omegaC, omegaB, hub, ns, np.log10(As), zre]
    data_in = np.concatenate((vals, cls ))
    data_collect.append(data_in)

if combine_files:
    data_collect = np.vstack(( data_collect, existingF))

np.savetxt(fileOut, data_collect, fmt='%.6e')
np.savetxt('Data/Ell_Values.dat', ell_vals)
