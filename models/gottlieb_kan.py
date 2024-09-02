# Modified from https://github.com/seydi1370/Basis_Functions

import torch
import torch.nn as nn

def gottlieb(n, x, alpha):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * alpha * x
    else:
        return 2 * (alpha + n - 1) * x * gottlieb(n-1, x, alpha) - (alpha + 2*n - 2) * gottlieb(n-2, x, alpha)


class GottliebKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, use_layernorm):
        super(GottliebKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.use_layernorm = use_layernorm
        self.layernorm = nn.LayerNorm(output_dim)
        self.alpha = nn.Parameter(torch.randn(1))
        self.gottlieb_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.gottlieb_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Gottlieb basis functions
        gottlieb_basis = []
        for n in range(self.degree + 1):
            gottlieb_basis.append(gottlieb(n, x, self.alpha))
        gottlieb_basis = torch.stack(gottlieb_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Gottlieb interpolation
        y = torch.einsum("bid,iod->bo", gottlieb_basis, self.gottlieb_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        
        if (self.use_layernorm == True):
            y = self.layernorm(y)
        
        return y

class GottliebKAN(nn.Module):
    def __init__(
        self, 
        layers_hidden,
        spline_order=3,  
    ):
        super(GottliebKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        #self.drop = torch.nn.Dropout(p=0.1) # dropout
        
        # duplicate middle layers
        layers_hidden = [layers_hidden[0]] + sum([[x]*2 for x in layers_hidden[1:-1]], []) + [layers_hidden[-1]]
       
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:-1]):
            
            self.layers.append(
                GottliebKANLayer(
                    input_dim,
                    output_dim,
                    degree=spline_order,
                    use_layernorm = True
                )
            )
        
        # last layer without layer norm
        self.layers.append(
                GottliebKANLayer(
                    layers_hidden[-2],
                    layers_hidden[-1],
                    degree=spline_order,
                    use_layernorm = False
                )
            )
    
    def forward(self, x: torch.Tensor):
        #x = self.drop(x)
        x = x.view(-1, self.layers_hidden[0])
        for layer in self.layers: 
            x = layer(x)
        return x
        
'''class OriginalGottliebKAN(nn.Module):
    def __init__(self):
        super(OriginalGottliebKAN, self).__init__()
        self.trigkan1 = OriginalGottliebKANLayer(784, 64, 3)
        self.bn1 = nn.LayerNorm(64)
        self.trigkan2 = OriginalGottliebKANLayer(64, 64, 3)
        self.bn2 = nn.LayerNorm(64)
        self.trigkan3 = OriginalGottliebKANLayer(64, 10, 3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.trigkan1(x)
        x = self.bn1(x)
        x = self.trigkan2(x)
        x = self.bn2(x)
        x = self.trigkan3(x)
        return x'''