
# Update

## 2 May, 2025
+ In the process of updating the code for submission to **Expert Systems with Applications**.

## 24 Feb, 2025
+ Add "quadratic3", x_1^2 + x_2^2 + ... + x_n^2
## 06 Dec, 2024 
+ Add SKAN and LArctan-SKAN: https://github.com/chikkkit/SKAN/, https://github.com/chikkkit/LArctan-SKAN
+ Add "attention" combination

# FC-KAN
In this repository, we apply function combinations in low-dimensional data to design Kolmogorov-Arnold Networks, referred to as **FC-KAN** (**F**unction **C**ombinations in **K**olmogorov-**A**rnold **N**etworks). The experiments demonstrate that these combinations improve the model performance.

Our paper, "FC-KAN: Function Combinations in Kolmogorov-Arnold Networks," is available at https://arxiv.org/abs/2409.01763 or https://www.researchgate.net/publication/383659216_FC-KAN_Function_Combinations_in_Kolmogorov-Arnold_Networks. 

<img src="https://github.com/hoangthangta/FC_KAN/blob/main/fc_kan_diagram.png" width="700" />

<img src="https://github.com/hoangthangta/FC_KAN/blob/main/train_losses.png" width="700" />

# How to combine?
We can use some element-wise operations to combine the functions' outputs by different methods.
```
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
        
        elif (self.combined_type == 'quadratic3'): 
            output = X[0, :, :].squeeze(0)*X[0, :, :].squeeze(0)
            for i in range(1, X.shape[0]):
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
```

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

# Training

## Parameters
* *mode*: working mode ("train" or "test"). Note that we did not write the test() function. =))
* *ds_name*: dataset name ("mnist" or "fashion_mnist").
* *model_name*: type of models (*bsrbf_kan*, *efficient_kan*, *fast_kan*, *faster_kan*, *mlp*, and *fc_kan*).
* *epochs*: the number of epochs.
* *batch_size*: the training batch size (default: 64).
* *n_input*: The number of input neurons (default: 28^2 = 784).
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (run.py) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST and Fashion-MNIST, there are 10 classes.
* *grid_size*: The size of grid (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu" (default: "cuda").
* *n_examples*: the number of examples in the training set used for training (default: 0, mean use all training data)
* *note*: A note saved in the model name file.
* *n_part*: the part of data used to train data (default: 0, mean use all training data, 0.1 means 10%).
* *func_list*: the name of functions used in FC-KAN (default='dog,rbf'). Other functions are *bs* and *base*, and functions in SKAN ('shifted_softplus', 'arctan', 'relu', 'elu', 'gelup', 'leaky_relu', 'swish', 'softplus', 'sigmoid', 'hard_sigmoid', 'sin', 'cos'). 
* *combined_type*: the type of data combination used in the output (default='quadratic', others are *sum*, *product*, *sum_product*, *concat*, *max*, *min*, *mean*). **We are developing other combinations.**
  
## Commands
See **run.sh** or **run_fc.sh** (```bash run.sh``` or ```bash run_fc.sh``` in BASH) for details.  We trained the models **on GeForce RTX 3060 Ti** (with other default parameters). For example, FC-KAN models (Difference of Gaussians + B-splines) can be trained on MNIST with different output combinations.

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "sum"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "product"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "sum_product"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "quadratic"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "concat"```

# References
* https://github.com/hoangthangta/BSRBF_KAN
* https://github.com/Blealtan/efficient-kan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/
* https://github.com/zavareh1/Wav-KAN
* https://github.com/seydi1370/Basis_Functions
* https://github.com/KindXiaoming/pykan (the original KAN)

# Paper
Cite our work if this paper is helpful for you.
```
@misc{ta2024fckan,
    title={FC-KAN: Function Combinations in Kolmogorov-Arnold Networks},
    author={Hoang-Thang Ta and Duy-Quy Thai and Abu Bakar Siddiqur Rahman and Grigori Sidorov and Alexander Gelbukh},
    year={2024},
    eprint={2409.01763},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

# Acknowledgement 
Also, give me a star if you like this repo. Thanks!

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
  
