import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from Network import *


from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
#rc('text', usetex=True)
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15

path = os.getcwd()

import warnings
#warnings.simplefilter("ignore", Warning)

def accuarcy_plot(LCDM=True, Nbranes=1, eCDM=0):

    
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    if LCDM:
        figname = 'plots/Accuracy_plot_LCDM_TT_.pdf'
        dataF = 'Data/LCDM_TT_Cls.dat'
        n_input = 6
        metaF = 'MetaGraphs/LCDM_TT_Cls/LCDM_TT_Cls_Graph_Global_'
    else:
        figname = 'plots/Accuracy_plot_MultiB_TT_Nbr_{:.1e}_Ecdm_{:.3e}.pdf'.format(Nbranes, eCDM)
        dataF = 'Data/MultiB_Nbr_{:.1e}_Ecdm_{:.3e}_TT_Cls.dat'.format(Nbranes, eCDM)
        n_input = 6
        metaF = 'MetaGraphs/MultiB_TT_Nbranes_{:.0f}_eCDM_{:.0f}_'.format(Nbranes, eCDM) + '/MultiB_TT_CLs_Graph_'

    dataL = np.loadtxt(dataF)
    inputs = dataL[:,:n_input]
    outputs = np.power(10, dataL[:, n_input:])

    NNet = ImportGraph(metaF, dataF)
    predict = np.power(10, NNet.run_yhat(inputs))
    
    plt.plot(outputs, predict, 'bo', alpha=0.1, ms=2)
    
    xvals = np.linspace(1, 1e4, 100)

    plt.plot(xvals, xvals, 'r', lw=1)
    plt.fill_between(xvals, xvals*(1. + 0.1), xvals*(1. - 0.1), color='r', alpha=0.2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylim([1, 1e4])
    plt.xlim([1, 1e4])
    fig.set_tight_layout(True)
    pl.savefig(figname)

    return
