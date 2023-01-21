import torch
from torch import nn

class ReTanh(torch.nn.Module):
    '''
    Rectified Hyperbolic Tangent
    '''
    def __init__(self, alpha=1.000000):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        beta = self.alpha / torch.tanh(torch.ones(1, dtype=torch.float64))
        return torch.fmax(torch.zeros(x.shape, dtype=torch.float64), beta * torch.tanh(x))