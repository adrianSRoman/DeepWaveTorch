"""Back projection layer
Project frequency domain mic correlation matrix into an image form of N pixels
"""

import torch
from torch import nn

class BackProjLayer(torch.nn.Module):
    """Spherical Convolutional Neural Netork.
    """

    def __init__(self, Nch, Npx, tau=None, D=None):
        """Initialization.
        Args:
            Nch (int): number of channels in mic array
            Npx (int): number of pixels in Robinson projection
        """
        super().__init__()
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
            

    def reset_parameters(self):
        std = 1e-5
        self.tau.data.normal_(0, std)
        self.D.data.normal_(0, std)

    def forward(self, S):
        """Forward Pass.
        Args:
            S (:obj:`torch.Tensor`): input to be forwarded. (N_sample, Npx)
        Returns:
            :obj:`torch.Tensor`: output: (N_sample, Npx)
        """
        N_sample, N_px = S.shape[0], self.tau.shape[0]
        y = torch.zeros((N_sample, N_px))
        for i in range(N_sample): # Loop to handle linalg.eigh: broadcasting can be slower
            Ds, Vs = torch.linalg.eigh(S[i]) # (Nch, Nch), (Nch, Nch)
            idx = Ds > 0  # To avoid np.sqrt() issues.
            Ds, Vs = Ds[idx], Vs[:, idx]
            y[i] = torch.linalg.norm(self.D.conj().T @ (Vs * torch.sqrt(Ds)), axis=1) ** 2 # (Npx, Nch) dot ((Nch, Nch) * (Nch, Nch))
        y -= self.tau
        return y