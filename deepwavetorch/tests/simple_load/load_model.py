import torch
import numpy as np

from deepwave.models.deepwave import DeepWave

N_px = 2234
R = np.load("./grid.npy")
K = 22
N_antenna = 48

deepwave_torch = DeepWave(R=R, kernel_size=K, 
                          Nch=N_antenna, Npx=N_px,
                          batch_size=1, depth=1,
                          pretr_params=None)

print(deepwave_torch)
