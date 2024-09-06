
# FC-KAN
In this repository, we apply function combinations in low-dimensional data to design Kolmogorov-Arnold Networks, referred to as **FC-KAN** (**F**unction **C**ombinations in **K**olmogorov-**A**rnold **N**etworks). The experiments demonstrate that these combinations improve the model performance.

Our paper "FC-KAN: Function Combinations in Kolmogorov-Arnold Networks": https://arxiv.org/abs/2409.01763

![The logarithmic values of training losses for the models over 25 epochs on MNIST and 35 epochs on
Fashion-MNIST. A quadratic function is used to combine B-Splines and DoG at the output of FC-KAN.](https://github.com/hoangthangta/FC_KAN/blob/main/train_losses.png)


# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

# Training

## Parameters
* *mode*: working mode ("train" or "test").
* *ds_name*: dataset name ("mnist" or "fashion_mnist").
* *model_name*: type of model (bsrbf_kan, efficient_kan, fast_kan, faster_kan).
* *epochs*: the number of epochs.
* *batch_size*: the training batch size.
* *n_input*: The number of input neurons.
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (run.py) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST, there are 10 classes.
* *grid_size*: The size of grid (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu".
* *n_examples*: the number of examples in the training set used for training (default: 0, mean use all training data)
* *note*: A note that is saved in the model name file
* *n_part*: the part of data used to train data (default: 0, mean use all training data, 0.1 means 10%)
* *func_list*: the name of functions used in FC-KAN (default='dog,rbf'). Other functions are *bs* and *base*.
* *combined_type*: the type of data combination used in the output (default='quadratic', others are *sum*, *product*, *sum_product*, *concat*, *max*, *min*, *mean*).
  
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
Give me a star if you like this repo. Thanks!

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
  
