# Original from:
#https://github.com/chikkkit/SKAN/blob/main/skan/skans.py
#https://github.com/chikkkit/LArctan-SKAN/blob/main/LArctan_SKAN_30epoch_lr000101.py

import torch
import numpy as np
# sin, cos, arctan, relu, leaky relu, swish, mnish, elu, softplus, sigmoid, hard_sigmoid, gelup

import torch
import numpy as np


def lsin(x, k):
    return k * torch.sin(x)

def lcos(x, k):
    return k * torch.cos(x)

def larctan(x, k):
    return k * torch.atan(x)

def lshifted_softplus(x, k):
    return torch.log(1 + torch.exp(k*x)) - np.log(2)
    
# y = max(0, x), ReLU
def lrelu(x, k):
    return torch.clamp(k*x, min=0)

# y = max(kx, x), leaky ReLU
def lleaky_relu(x, k):
    return torch.max(k*x, x)

# y = x / (1 + e^(-kx)), Swish
def lswish(x, k):
    return x / (1 + torch.exp(-k*x))

# y = x / (1 + e^(-kx)), Mish
def lmish(x, k):
    return x * torch.tanh(torch.log(1 + torch.exp(k*x)))

# y = log(1 + e^(kx)), Softplus
def lsoftplus(x, k):
    return torch.log(1 + torch.exp(k*x))

# y = max(0, min(1, (kx + 0.5))), Hard sigmoid
def lhard_sigmoid(x, k):
    return torch.clamp(k*x + 0.5, min=0, max=1)

# y = k * (e^(x/k) - 1) if x < 0, else x, Exponential Linear Unit (ELU)
def lelu(x, k):
    return torch.where(x < 0, k * (torch.exp(x/k) - 1), x)

# y = log(1 + e^(kx)) - log(2), Shifted Softplus
def lshifted_softplus(x, k):
    return torch.log(1 + torch.exp(k*x)) - np.log(2)

# y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (kx + 0.044715 * k^3 * x^3))), Gaussian Error Linear Unit with Parameter (GELU-P)
def lgelup(x, k):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2/np.pi) * (k*x + 0.044715 * k**3 * x**3)))

'''
def lsin(x):
    return torch.sin(x)

def lcos(x):
    return torch.cos(x)
    
def larctan(x):
    return torch.atan(x)

# ReLU
def lrelu(x):
    return torch.clamp(x, min=0)
    
# Leaky ReLU
def lleaky_relu(x, k = 0.01): 
    return torch.max(k*x, x)

# Swish
def lswish(x):
    return x / (1 + torch.exp(-x))

# Mish
def lmish(x):
    return x * torch.tanh(torch.log(1 + torch.exp(x)))

# Softplus
def lsoftplus(x):
    return torch.log(1 + torch.exp(x))

# Hard sigmoid
def lhard_sigmoid(x):
    return torch.clamp(x + 0.5, min=0, max=1)

# Sigmoid function
def lsigmoid(x):
    return 1 / (1 + torch.exp(-x))
    
# Exponential Linear Unit (ELU)
def lelu(x):
    return torch.where(x < 0, (torch.exp(x) - 1), x)
    
# Shifted Softplus
def lshifted_softplus(x):
    return torch.log(1 + torch.exp(x)) - np.log(2)
    
# Gaussian Error Linear Unit with Parameter (GELU-P)
def lgelup(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))'''