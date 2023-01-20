"""Chebyshev convolution layer. 
PyTorch implementation inspired from: https://github.com/deepsphere/deepsphere-pytorch/blob/master/deepsphere/layers/chebyshev.py
Based upon NumPy implementation from: https://github.com/imagingofthings/DeepWave
"""
# pylint: disable=W0221

import math

import torch
from torch import nn

def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere. (Npx, Npx)
        inputs (:obj:`torch.Tensor`): The current input data being forwarded. (Nsamps, Npx)
        weight (:obj:`torch.Tensor`): The weights of the current layer. (K,)
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution. (Nsamps, Npx)  
    """
    K = weight.shape[0]
    was_1d = (inputs.ndim == 1)
    if was_1d:
        inputs = inputs.unsqueeze(0)
    N_sample, Npx = inputs.shape[0], inputs.shape[1]

    x0 = inputs.T # (Nsamps, Npx).T -> (Npx, Nsamps)
    inputs = x0.T.unsqueeze(0) # (1, Npx, Nsamps)
    x1 = torch.sparse.mm(laplacian, x0) # (Npx, Npx) x (Npx, Nsamps) -> (Npx, Nsamps)
    inputs = torch.cat((inputs, x1.T.unsqueeze(0)), 0) # (1, Nsamps, Npx) + (1, Nsamps, Npx) = (2*Nsamps, Npx)
    for _ in range(1, K - 1):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0 # (Npx, Npx) x (Npx, Nsamps) - (Npx, Nsamps) = (Npx, Nsamps)
        inputs = torch.cat((inputs, x2.T.unsqueeze(0)), 0)  # (ki, Nsamps, Npx) + (1, Nsamps, Npx) 
        x0, x1 = x1, x2 # (Npx, Nsamps), (Npx, Nsamps)
    inputs = inputs.permute(1, 2, 0).contiguous()  # (K, Nsamps, Npx)
    inputs = inputs.view([N_sample*Npx, K])  # (Nsamps*Npx, K)
    inputs = inputs.matmul(weight)  # (Nsamps*Npx, K) x (K,) -> (Nsamps*Npx,)
    inputs = inputs.view([N_sample, Npx]) # (Nsamps, Npx)                     
    return inputs

class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, weight=None, bias=False, conv=cheb_conv):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            weight (torch.Tensor): pre-trained or intial state weight matrix (K,)
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the graph convolution.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv
        if weight is None:
            shape = (kernel_size,)
            self.weight = torch.nn.Parameter(torch.DoubleTensor(*shape))
            std = math.sqrt(2 / (self.in_channels * self.kernel_size))
            self.weight.data.normal_(0, std)
        else:
            self.weight = torch.nn.Parameter(weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.DoubleTensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.bias_initialization()

    def bias_initialization(self):
        """Initialize bias.
        """
        if self.bias is not None:
            self.bias.data.fill_(0.00001)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs
    
class SphericalChebConv(torch.nn.Module):
    """Chebyshev Graph Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size, weight=None):
        """Initialization.
        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.DoubleTensor`): laplacian
            kernel_size (int): order of polynomial filter K. Defaults to 3.
            weight (:obj:`torch.sparse.DoubleTensor`): weight convolutional matrix (K,)
        """
        super().__init__()
        self.register_buffer("laplacian", lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size, weight)

    '''
    def state_dict(self, *args, **kwargs):
        """! WARNING !
        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if key.endswith("laplacian"):
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict
    '''

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]
        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.chebconv(self.laplacian, x)
        return x