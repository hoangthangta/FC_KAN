# Original from:
#https://github.com/chikkkit/SKAN/blob/main/skan/skans.py
#https://github.com/chikkkit/LArctan-SKAN/blob/main/LArctan_SKAN_30epoch_lr000101.py

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from models.functions import *

class SKANLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, basis_function='shifted_softplus', device='cpu'):
        super(SKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        self.basis_function = self.set_basis_function(basis_function)
        self.device = device
        
        # add bias
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features+1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()
        
        self.layernorm = nn.LayerNorm(in_features)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
    def set_basis_function(self, function = 'shifted_softplus'):
        if (function == 'shifted_softplus'):
            return lshifted_softplus
        elif (function == 'arctan'):
            return larctan
        elif (function == 'cos'):
            return lcos
        elif (function == 'sin'):
            return lsin
        elif (function == 'relu'):
            return lrelu
        else:
            # check functions.py to get more functions
            raise Exception('The function "' + function + '" does not support!')
    
    def forward(self, x):
        x = self.layernorm(x)  # different from original code
        
        x = x.view(-1, 1, self.in_features)
        # add bias
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)
        y = self.basis_function(x, self.weight)
        
        y = torch.sum(y, dim=2)
        return y
    
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
class SKAN(nn.Module):
    def __init__(self, layer_sizes, basis_function='shifted_softplus', bias=True, device='cpu'):
        super(SKAN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes)-1):
            self.layers.append(SKANLinear(layer_sizes[i], layer_sizes[i+1], bias=bias, 
                                             basis_function=basis_function, device=device))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x