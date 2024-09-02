import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_activation = torch.nn.SiLU,

    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.output_dim = output_dim
        self.base_activation = base_activation()
        self.input_dim = input_dim
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        #self.drop = nn.Dropout(p=0.01) # dropout
 
    def forward(self, x):
        
        #x = self.drop(x)
        
        # layer normalization
        x = self.layernorm(x)
        
        base_output = self.base_activation(F.linear(x, self.base_weight))

        return base_output 

class MLP(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        base_activation=torch.nn.SiLU,
    ):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        #self.drop = torch.nn.Dropout(p=0.1) # dropout
        
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MLPLayer(
                    input_dim,
                    output_dim,
                    base_activation=base_activation,
                )
            )
    
    def forward(self, x: torch.Tensor):
        
        for layer in self.layers: 
            x = layer(x)
            #x = self.drop(x)
        #x = F.log_softmax(x, dim=1)
        return x
        
