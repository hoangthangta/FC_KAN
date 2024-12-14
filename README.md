
# Update
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
  
