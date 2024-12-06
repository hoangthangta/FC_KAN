import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functions import *

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -1.5,
        grid_max: float = 1.5,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
        
        
class FC_KANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        func_list: list,
        grid_size = 5,
        spline_order = 3,
        base_activation = torch.nn.SiLU,
        grid_range=[-1.5, 1.5],
        bias = False

    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.base_activation = base_activation()
        self.input_dim = input_dim
        self.func_list = func_list
        self.use_bias = bias # for SKAN
        
        # add bias
        if bias:
            #self.weight = nn.Parameter(torch.Tensor(out_features, in_features+1).to(device))
            self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim + 1))
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        else:
            #self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
            self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
                
        self.spline_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim*(grid_size+spline_order)))
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size+spline_order)
        
        h = (grid_range[1] - grid_range[0]) / grid_size # 0.45, 0.5
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h 
                + grid_range[0]
            )
            .expand(self.input_dim, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        #self.linear = nn.Linear(self.input_dim*(grid_size+spline_order), self.output_dim)
        #self.drop = nn.Dropout(p=0.1) # dropout
        
        self.scale = nn.Parameter(torch.ones(self.output_dim, self.input_dim))
        self.translation = nn.Parameter(torch.zeros(self.output_dim, self.input_dim))
        
        # Batch normalization
        #self.bn = nn.BatchNorm1d(self.output_dim)
        
        
    def b_splines(self, x: torch.Tensor):
        """
            Compute the B-spline bases for the given input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            Returns:
                torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim

        grid: torch.Tensor = (
            self.grid
        )  
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        assert bases.size() == (
            x.size(0),
            self.input_dim,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()    

    def wavelet_transform(self, x, wavelet_type = 'dog'):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded
            
        # Implementation of different wavelet types
        if wavelet_type == 'mexh':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet 
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            #wavelet_output = wavelet_weighted
                
        elif wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2,torch.ones_like(v),torch.where(v >= 1,torch.zeros_like(v),torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            #You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output  
 
    def forward(self, X):
        
        device = X.device
        
        # Layer normalization
        X = self.layernorm(X)
        
        output = torch.zeros(X.shape[0], X.shape[1], self.output_dim).to(device)
        for i, f in zip(range(X.shape[0]), self.func_list):
            
            x = X[i, :, :].squeeze(0)
            if (f == 'rbf'):
                x = self.rbf(x).view(x.size(0), -1)
                x = F.linear(x, self.spline_weight)
            elif (f == 'bs'):
                x = self.b_splines(x).view(x.size(0), -1)
                x = F.linear(x, self.spline_weight)
            elif (f == 'dog'):
                x = self.wavelet_transform(x, wavelet_type = 'dog')
            elif (f == 'base'):
                x = F.linear(self.base_activation(x), self.base_weight)
            
            elif (f in ['shifted_softplus', 'arctan', 'relu', 'elu', 'gelup', 'leaky_relu', 'swish', 'softplus', 'sigmoid', 'hard_sigmoid', 'sin', 'cos']):
                x = x.view(-1, 1, self.input_dim)
                if self.use_bias:
                    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)   
                
                if (f == 'shifted_softplus'): # shifted softplus
                    x = llshifted_softplus(x, self.base_weight)
                elif (f == 'arctan'):
                    x = larctan(x, self.base_weight)
                elif (f == 'relu'):
                    x = lrelu(x, self.base_weight)
                elif (f == 'elu'):
                    x = lelu(x, self.base_weight)
                elif (f == 'gelup'):
                    x = lgelup(x, self.base_weight)
                elif (f == 'leaky_relu'):
                    x = lleaky_relu(x, self.base_weight)
                elif (f == 'swish'):
                    x = lswish(x, self.base_weight)
                elif (f == 'softplus'):
                    x = lsoftplus(x, self.base_weight)
                elif (f == 'sigmoid'):
                    x = lsigmoid(x, self.base_weight)
                elif (f == 'hard_sigmoid'):
                    x = lhard_sigmoid(x, self.base_weight)
                elif (f == 'sin'):
                    x = lsin(x, self.base_weight)
                elif (f == 'cos'):
                    x = lcos(x, self.base_weight)
                x = torch.sum(x, dim=2)
            else:
                raise Exception('The function "' + f + '" does not support!')
                # Write more functions here...
            output[i] = x
        
        return output

class FC_KAN(torch.nn.Module):
        
    def __init__(
        self, 
        layer_list,
        func_list,
        grid_size=5,
        spline_order=3,  
        combined_type = 'quadratic',
        #output_type = 'all',
        base_activation=torch.nn.SiLU,
    ):
        super(FC_KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        self.func_list = func_list
        self.combined_type = combined_type
        #self.output_type = output_type
        #self.drop = torch.nn.Dropout(p=0.1) # dropout
        self.base_activation = base_activation()
        
        self.concat_weight = torch.nn.Parameter(torch.Tensor(layer_list[-1], len(func_list)*layer_list[-1]))
        torch.nn.init.kaiming_uniform_(self.concat_weight, a=math.sqrt(5))
        
        for input_dim, output_dim in zip(layer_list, layer_list[1:]):
            self.layers.append(
                FC_KANLayer(
                    input_dim,
                    output_dim,
                    func_list,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                )
            )
    
    def combine_attention(self, x_set):
        """
        Combine a set of tensors using an attention mechanism.
        
        Args:
            x_set (torch.Tensor): Tensor of shape (n, batch_size, feature_dim).
                                  n = number of tensors.
        
        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, feature_dim).
        """
        n, batch_size, feature_dim = x_set.shape

        # Compute pairwise attention scores
        # Flatten tensors along batch and feature dimensions
        queries = x_set.view(n, -1)  # Shape: (n, batch_size * feature_dim)
        keys = x_set.view(n, -1).T  # Shape: (batch_size * feature_dim, n)

        attention_scores = F.softmax(torch.matmul(queries, keys), dim=1)  # Shape: (n, n)

        # Apply attention scores to combine tensors
        # Expand dimensions for broadcasting
        attention_scores = attention_scores.unsqueeze(-1).unsqueeze(-1)  # Shape: (n, n, 1, 1)
        weighted_tensors = attention_scores * x_set.unsqueeze(0)  # Shape: (n, n, batch_size, feature_dim)

        # Sum and prod over the set dimension
        combined_tensor = torch.sum(weighted_tensors, dim=0)
        combined_tensor = torch.prod(combined_tensor, dim=0)
        
        return combined_tensor
        
    def forward(self, x: torch.Tensor):
        #x = self.drop(x)
        #device = x.device
        
        if (len(self.func_list) == 1):
            raise Exception('The number of functions (func_list) must be larger than 1.')
        
        X = torch.stack([x] * len(self.func_list)) # size (number of functions, batch_size, input_dim)
        for layer in self.layers: 
            X = layer(X)
        
        output = X.detach().clone()
        if (self.combined_type == 'sum'): output = torch.sum(X, dim=0)
        elif (self.combined_type == 'product'):  
            '''
            # Use only for very large tensors. This is slower and can have cumulative numerical errors
            output_prod = torch.ones(X.shape[1:], device=X.device)
            for i in range(X.shape[0]):
                output_prod *= X[i, :, :]
            '''
            output = torch.prod(X, dim=0)
        elif (self.combined_type == 'sum_product'): output = torch.sum(X, dim=0) +  torch.prod(X, dim=0)
        elif (self.combined_type == 'quadratic'): 
            output = torch.sum(X, dim=0) +  torch.prod(X, dim=0) 
            for i in range(X.shape[0]):
                output = output + X[i, :, :].squeeze(0)*X[i, :, :].squeeze(0)
            #output += torch.sum(X ** 2, dim=0) # can lead to memory error
            
        elif (self.combined_type == 'quadratic2'): 
            output = torch.prod(X, dim=0) 
            for i in range(X.shape[0]):
                output = output + X[i, :, :].squeeze(0)*X[i, :, :].squeeze(0)
            #output += torch.sum(X ** 2, dim=0) # can lead to memory error
            
        elif (self.combined_type == 'cubic'):
            outsum = torch.sum(X, dim=0)
            output = outsum +  torch.prod(X, dim=0) 
            for i in range(X.shape[0]):
                output = output + X[i, :, :].squeeze(0)*X[i, :, :].squeeze(0)
            output = output*outsum
            
        elif (self.combined_type == 'concat'):
            X_permuted = X.permute(1, 0, 2)
            output = X_permuted.reshape(X_permuted.shape[0], -1)
        elif (self.combined_type == 'concat_linear'):
            X_permuted = X.permute(1, 0, 2)
            output = X_permuted.reshape(X_permuted.shape[0], -1)
            output = F.linear(output, self.concat_weight)
        elif (self.combined_type == 'max'):
            output, _ = torch.max(X, dim=0)
        elif (self.combined_type == 'min'):
            output, _ = torch.min(X, dim=0)
        elif (self.combined_type == 'mean'):
            output = torch.mean(X, dim=0)
        elif (self.combined_type == 'rbf'): # for fun only but it is good on MNIST 

            z = torch.zeros(X.shape[1], X.shape[2]).to(device)
            gamma = 0.5

            # Combine all pairs of tensors in X using RBFs
            n = len(self.func_list)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        z += torch.exp(-gamma * torch.pow(X[i] - X[j], 2))

            # Optionally, normalize by the number of combinations
            z /= n * (n - 1)
            output = z*X.shape[2] # normalize to the range of output values
        elif (self.combined_type == 'attention'):
            output = self.combine_attention(X)
        else:
            raise Exception('The combined type "' + self.combined_type + '" does not support!')
            # Write more combinations here...

        #output = self.base_activation(output) # SiLU
        #output = output + F.normalize(output, p=2, dim=1)

        return output
