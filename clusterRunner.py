import numpy as np
import os
from CMB_c import CMB
import multiprocessing as mp
from constants import *
import itertools
import time
t_0 = time.time()

Multiverse = True
Nbranes = 1e0
PressureFac = 1e-6
extraCDM = 0.000

HubbleParam = 67.66
n_s_index = 0.9619
A_s_norm = 2.215e-9
Neff = 3.046
z_reion = 10.
Om_b_load = 0.0492
Om_c_load = 0.266

# Modify rho_c
rho_critical *= (HubbleParam / 100.)**2.

if not Multiverse:
    Ftag = 'StandardUniverse'
    OM_b = Om_b_load
    OM_c = Om_c_load
    OM_g = 5.43e-5
    OM_L = 0.7 # Doesnt matter, it calculates with flat Uni

    OM_b2 = 0.
    OM_c2 = 0.
    OM_g2 = 0.
    OM_L2 = 0.
else:
    Ftag = 'MultiBrane'
    omega_cdm = Om_c_load
    
    OM_b = Om_b_load
    OM_c = 0.
    OM_g = 5.43e-5
    OM_L = 0.

    if extraCDM == 0:
        OM_b2 =  omega_cdm / Nbranes
        OM_c2 = 0.
        OM_g2 = PressureFac * (OM_g / OM_b) * OM_b2
        OM_L2 = 0.
    else:
        OM_c = extraCDM 
        OM_c2 = 0.
        OM_b2 = (omega_cdm - extraCDM) / Nbranes
        OM_g2 = PressureFac * (OM_g / OM_b) * OM_b2
        OM_L2 = 0.

lmax_Pert = 7
process_Num = 10

compute_LP = True
compute_CMB = True
compute_MPS = False
# Note, don't copute MPS and CMB at same time. This requires different kgrid...

if compute_MPS:
    kmin = 5e-4
    kmax = 15.
    knum = 60
else:
    kmin = 8e-4
    kmax = 4e-1
    knum = 120

kTotNum = 5000
lmax = 2500
kgrid_Full = np.logspace(np.log10(kmin), np.log10(kmax), kTotNum)

if compute_MPS:
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), knum)
else:
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), knum)

SetCMB = CMB(OM_b, OM_c, OM_g, OM_L, kmin=kmin, kmax=kmax, knum=kTotNum, lmax=lmax,
             lvals=1, Ftag=Ftag, lmax_Pert=lmax_Pert, multiverse=Multiverse,
             OM_b2=OM_b2, OM_c2=OM_c2, OM_g2=OM_g2, OM_L2=OM_L2,
             HubbleParam=HubbleParam, n_s_index=n_s_index, A_s_norm=A_s_norm,
             z_reion=z_reion, Neff=Neff, Nbrane=Nbranes, killF=True)


def CMB_wrap(kval):
    try:
        chk = len(kval)
        klist = kval
    except:
        klist = [kval]
    returnlist = np.zeros(len(klist), dtype=object)
    for i,kv in enumerate(klist):
        indx = np.where(kv == kgrid)[0][0]
        hold = SetCMB.runall(kv, indx, compute_LP=compute_LP, compute_TH=False,
                compute_CMB=True, compute_MPS=False)
#        print('Finsihed Computing k: ', kv)
        if hold is None:
            returnlist[i] = [indx, False]
        else:
            returnlist[i] = [indx, hold]
    return returnlist


ThetaTabTot = np.zeros_like(SetCMB.ThetaTabTot)
mps_tks = np.zeros((len(kgrid), 2))
kfail = []
ellTab = SetCMB.ThetaTabTot[0, :]
ThetaTabTot[0,:] = ellTab

def runPl(kgrid, processes=1):

    if __name__ == '__main__':
        pool = mp.Pool(processes=processes)
        hold_info = pool.map_async(CMB_wrap, kgrid)
        structure = hold_info.get()
        pool.close()
    flat_list = [item for sublist in structure for item in sublist]
    return flat_list

def get_ThetaTab(x):
    integ2 = [S_SW[:, i] * SetCMB.sphB[x](listC[i*len(tau_list):len(tau_list)*(i+1)]) +
              S_D[:, i] * SetCMB.sphB_D[x](listC[i*len(tau_list):len(tau_list)*(i+1)]) + S_ISW[:, i] *
              SetCMB.sphB[x](listC[i*len(tau_list):len(tau_list)*(i+1)]) for i in range(len(kgrid_Full))]
    return [np.trapz(integ2[i], tau_list) for i in range(len(integ2))]

structure = []
if compute_LP:
    print('Running Pool...')
    t_k = time.time()
    structure = runPl(kgrid, processes=process_Num)
    list_ord = []
    t_k_end = time.time()
    print('Time to Compute k-evolution: ', t_k_end - t_k)
    for i in range(len(structure)):
        list_ord.append(structure[i][0])
        structure[i] = structure[i][1]
    structure = np.asarray(structure)[list_ord]


    if compute_CMB:
        print('Interpolating Sources and Calculating Bessel Functions...')
        t_k = time.time()
        tau_list, S_SW, S_D, S_ISW = SetCMB.theta_construction(structure, kgrid)

        t_k_end = time.time()
        print('Time to Compute Theta Construction: ', t_k_end - t_k)
        listC = [x*y for x,y in list(itertools.product(kgrid_Full, (SetCMB.eta0 - tau_list)))]
        par_scan = range(len(ellTab))
        if __name__ == '__main__':
            pool = mp.Pool(processes=process_Num)
            structureTheta = pool.map(get_ThetaTab, par_scan)
            pool.close()
        t_k = time.time()
        print('Time to Compute Theta Table: ',  t_k - t_k_end)
        for i in range(len(structureTheta)):
            SetCMB.ThetaTabTot[1:, i] = structureTheta[i]

        SetCMB.computeCMB(kgrid_Full, SetCMB.ThetaTabTot)
        t_k_end = time.time()
        print('Time to Compute Cls: ',  t_k_end - t_k)
    else:
        for i in range(len(structure)):
            if isinstance(structure[i][1], bool):
                kfail.append(structure[i][0])
            else:
                mps_tks[structure[i][0], 0] = kgrid[structure[i][0]]
                mps_tks[structure[i][0], 1] = structure[i][1]

if compute_MPS:
    print('Kfail: ', kgrid[kfail], len(kgrid[kfail]))
    for i in range(len(kgrid))[::-1]:
        if i in kfail:
            kgrid = np.delete(kgrid, [i], axis=0)
            mps_tks = np.delete(mps_tks, [i], axis=0)
    SetCMB.runall(1, 1, compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=compute_MPS, ThetaTab=mps_tks)


print(time.time() - t_0)
