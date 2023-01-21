import numpy as np
import deepwave.nn as nn
import deepwave.tools.math.func as func
import deepwave.tools.math.graph as graph
import deepwave.nn.crnn as deepwave_numpy # NumPy DeepWave model library

import torch
from deepwavetorch.models.deepwave import DeepWave as deepwave_torch # PyTorch DeepWave model

Df = nn.DataSet.from_file("/home/asroman/repos/DeepWave/datasets/FRIDA/dataset/D_1-5_freq0_cold.npz") # 0th frequency training data
Pf = np.load("/home/asroman/repos/DeepWave/datasets/FRIDA/dataset/D_freq0_train.npz") # trained model parameters
N_antenna = Df.XYZ.shape[1] # number of microphones in the array data
N_px = Df.R.shape[1] # number of pixels in the intensity map
K = int(Pf['K']) # Chev filter polynomial order
print("Num antenna:", N_antenna)
print("Num pixels:", N_px)
print("Filter Kth:", K)
parameter = deepwave_numpy.Parameter(N_antenna, N_px, K)
sampler = Df.sampler()
p_opt = Pf['p_opt'][np.argmin(Pf['v_loss'])]

# We will use the 0th image to test DeepWave original VS. DeepWave Pytorch
S, I, I_prev = sampler.decode(Df[0])
N_layer = Pf['N_layer']
print("Number of layers:", N_layer)
p_mu, p_D, p_tau = parameter.decode(p_opt) # Load trained parameters

# Load the DeepWave NumPy model
Ln, _ = graph.laplacian_exp(Df.R, normalized=True) 
afunc = lambda _: func.retanh(Pf['tanh_lin_limit'], _) 
deepwavenumpy = deepwave_numpy.Evaluator(N_layer, parameter, p_opt, Ln, afunc)

# Next we want to re-use the network parameters for the PyTorch DeepWave model
# Save orinal DeepWave network's trained parameters
pretr_params = {"mu": torch.from_numpy(p_mu), 
                "tau": torch.from_numpy(p_tau),
                "D": torch.from_numpy(p_D)}
# Load the DeepWave PyTorch model
deepwavetorch = deepwave_torch(R=Df.R, kernel_size=K, Nch=N_antenna, Npx=N_px,
                 batch_size=1, depth=1, pretr_params=pretr_params)

# Comparison of NumPy Vs. PyTorch DeepWave model
I_numpy = deepwavenumpy(S, I_prev)
I_torch = deepwavetorch(torch.from_numpy(S).unsqueeze(0), torch.from_numpy(I_prev).double()).cpu().detach().numpy()
print("Intensity fields MSE error (NumPy vs. PyTorch):")
mse = np.square(np.subtract(I_numpy, I_torch)).mean()
print(mse) # NOTE: we expect to see 1e-15 error, which is pretty close to machine epsilon



