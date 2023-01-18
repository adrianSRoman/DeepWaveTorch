import torch
from torch import nn

from deepwave.layers.backproj import BackProjLayer
from deepwave.layers.graph_conv import SphericalChebConv
from deepwave.utils.activations import ReTanh
from deepwave.utils.laplacian import laplacian_exp

class DeepWave(torch.nn.Module):
    """DeepWave: real-time recurrent neural network for acoustic imaging.
    """

    def __init__(self, R, kernel_size, Nch, Npx, batch_size=1, depth=1, pretr_params=None):
        """Initialization.
        Args:
            R: Cartesian coordinates of point set (N,3)
            kernel_size (int): polynomial degree.
            Nch (int): number of channels in mic array
            Npx (int): number of pixels in Robinson projection
        """
        super().__init__()
        
        self.laps, self.rho = laplacian_exp(R, depth)
        if pretr_params: # use pre-trained parameters: tau, mu and D
            self.y_backproj = BackProjLayer(Nch, Npx, tau=pretr_params['tau'], D=pretr_params['D'])
            self.sconvl = SphericalChebConv(Nch, Npx, self.laps[0], kernel_size, weight=pretr_params['mu'])
        else:
            self.y_backproj = BackProjLayer(Nch, Npx)
            self.sconvl = SphericalChebConv(Nch, Npx, self.laps[0], kernel_size)
        self.retanh = ReTanh(alpha=1.000000)

    def reset_parameters(self):
        std = 1e-4 
        self.I.data.random_(0, std)
        
    def forward(self, S, I_prev=None):
        """Forward Pass.
        Args:
            S (:obj:`torch.Tensor`): input (Nch, Nch)
        Returns:
            y_proj :obj: `torch.Tensor`: output (N_sample, Npx)
            x_conv* :obj: `torch.Tensor`: output (N_samle, Npx)
        """
        y_proj  = self.y_backproj(S)
        if I_prev is None:
            I_prev = torch.zeros(y_proj.shape[1], dtype=torch.float64)
        x_conv4 = self.sconvl(I_prev) 
        x_conv4 = x_conv4.add(y_proj)
        x_conv4 = self.retanh(x_conv4)
        x_conv3 = self.sconvl(x_conv4)
        x_conv3 = x_conv3.add(y_proj)
        x_conv3 = self.retanh(x_conv3)
        x_conv2 = self.sconvl(x_conv3)
        x_conv2 = x_conv2.add(y_proj)
        x_conv2 = self.retanh(x_conv2)
        x_conv1 = self.sconvl(x_conv2)
        x_conv1 = x_conv1.add(y_proj)
        x_conv1 = self.retanh(x_conv1)
        x_conv0 = self.sconvl(x_conv1)
        x_conv0 = x_conv0.add(y_proj)
        x_conv0 = self.retanh(x_conv0)
        out = x_conv0

        return out