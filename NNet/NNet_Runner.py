import os
from Network import *
import numpy as np

# Train NNet
Train = True
KeepTraining = False
# Load and Evaluate NNet
Eval = False

LCDM = False
Nbranes = 1
eCDM = 0.

epochs = 50000
HiddenNodes = 50

# log10 P,  om_b, om_c, H0, ns, As, zreion
vec_eval = [[-3, 0.04, 0.27, 70., 0.96, np.log10(3e-9), 10.]]

    
init_NN = MLP_Nnet(HiddenNodes=HiddenNodes, epochs=epochs, LCDM=LCDM, Nbranes=Nbranes, eCDM=eCDM)
init_NN.main_nnet()

if Train:
    init_NN.train_NN(vec_eval, keep_training=KeepTraining)
if Eval:
    init_NN.eval_NN(vec_eval)
