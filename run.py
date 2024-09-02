import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from models import EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, GottliebKAN
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from file_io import *

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def run(model_name = 'bsrbf_kan', batch_size = 64, n_input = 28*28, epochs = 10, n_output = 10, n_hidden = 64, \
        grid_size = 5, num_grids = 8, spline_order = 3, ds_name = 'mnist', n_examples = -1, note = 'full', n_part = 0.1, func_list = [], \
        combined_type = 'quadratic'):

    start = time.time()
    
    # Fashion-MNIST
    # Mean: 0.2860, Standard Deviation: 0.3530

    # MNIST
    # Mean: 0.1307, Standard Deviation: 0.3081
    
    # Sign Language MNIST
    # Mean: 0.6257, Standard Deviation: 0.1579
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
        
    trainset, valset = [], []
    if (ds_name == 'mnist'):
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        valset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif(ds_name == 'fashion_mnist'):
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )

        valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif(ds_name == 'sl_mnist'):
        from ds_model import SignLanguageMNISTDataset
        trainset = SignLanguageMNISTDataset(csv_file='data/SignMNIST/sign_mnist_train.csv', transform=transform)
        valset = SignLanguageMNISTDataset(csv_file='data/SignMNIST/sign_mnist_test.csv', transform=transform)

    if (n_examples > 0):
        if (n_examples/batch_size > 1):
            trainset = torch.utils.data.Subset(trainset, range(n_examples))
        else:
            print('The number of examples is too small!')
            return
    elif(n_part > 0):
        if (len(trainset)*n_part > batch_size):
            trainset = torch.utils.data.Subset(trainset, range(int(len(trainset)*n_part)))
        else:
            print('n_part is too small!')
            return

    print('trainset: ', len(trainset))
    print('valset: ', len(valset))
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Create model storage
    output_path = 'output/' + ds_name + '/' + model_name + '/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    saved_model_name, saved_model_history = '', ''
    if (model_name == 'fc_kan'):
        saved_model_name = model_name + '__' + ds_name + '__' + '-'.join(x for x in func_list) + '__' + combined_type + '__' + note + '.pth'
        saved_model_history = model_name + '__' + ds_name + '__' + '-'.join(x for x in func_list) + '__' + combined_type + '__' + note + '.json'
    else:
        saved_model_name = model_name + '__' + ds_name + '__' + note + '.pth'
        saved_model_history =  model_name + '__' + ds_name + '__' + note + '.json'
    with open(os.path.join(output_path, saved_model_history), 'w') as fp: pass

    # Define model
    model = {}
    print('model_name: ', model_name)
    if (model_name == 'bsrbf_kan'):
        model = BSRBF_KAN([n_input, n_hidden, n_output], grid_size = grid_size, spline_order = spline_order)
    elif(model_name == 'fast_kan'):
        model = FastKAN([n_input, n_hidden, n_output], num_grids = num_grids)
    elif(model_name == 'faster_kan'):
        model = FasterKAN([n_input, n_hidden, n_output], num_grids = num_grids)
    elif(model_name == 'gottlieb_kan'):
        model = GottliebKAN([n_input, n_hidden, n_output], spline_order = spline_order)
    elif(model_name == 'mlp'):
        model = MLP([n_input, n_hidden, n_output])
    elif(model_name == 'fc_kan'):
        model = FC_KAN([n_input, n_hidden, n_output], func_list, combined_type = combined_type)
    else:
        model = EfficientKAN([n_input, n_hidden, n_output], grid_size = grid_size, spline_order = spline_order)
    model.to(device)
    
    print('parameters: ', count_parameters(model))
    
    #return
    
    # Define optimizer
    lr = 1e-3
    wc = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wc)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    # Define loss
    criterion = nn.CrossEntropyLoss()

    best_epoch, best_accuracy = 0, 0
    y_true = [labels.tolist() for images, labels in valloader]
    y_true = sum(y_true, [])
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_accuracy, train_loss = 0, 0
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, n_input).to(device)
                optimizer.zero_grad()
                output = model(images.to(device))
                loss = criterion(output, labels.to(device))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                #accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                train_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                pbar.set_postfix(loss=train_loss/len(trainloader), accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])
        
        train_loss /= len(trainloader)
        train_accuracy /= len(trainloader)
            
        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        
        y_pred = []
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, n_input).to(device)
                output = model(images.to(device))
                val_loss += criterion(output, labels.to(device)).item()
                
                y_pred += output.argmax(dim=1).tolist()

                val_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
       
        # calculate F1, Precision and Recall
        #f1 = f1_score(y_true, y_pred, average='micro')
        #pre = precision_score(y_true, y_pred, average='micro')
        #recall = recall_score(y_true, y_pred, average='micro')
        
        f1 = f1_score(y_true, y_pred, average='macro')
        pre = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Update learning rate
        scheduler.step()
        
        # Choose best model
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model, output_path + '/' + saved_model_name)
              
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
        print(f"Epoch [{epoch}/{epochs}], Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}, F1: {f1:.6f}, Precision: {pre:.6f}, Recall: {recall:.6f}")
        
        write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'epoch':epoch, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'f1_macro':f1, 'pre_macro':pre, 're_macro':recall, 'best_epoch':best_epoch, 'val_loss': val_loss, 'train_loss':train_loss}, file_access = 'a')
    
    end = time.time()
    print(f"Training time (s): {end-start}")
    write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')
    
def main(args):
    
    func_list = args.func_list.split(',')
    func_list = [x.strip() for x in func_list]
    
    if (args.mode == 'train'):
        run(model_name = args.model_name, batch_size = args.batch_size, epochs = args.epochs, \
            n_input = args.n_input, n_output = args.n_output, n_hidden = args.n_hidden, \
            grid_size = args.grid_size, num_grids = args.num_grids, spline_order = args.spline_order, ds_name = args.ds_name, n_examples =  args.n_examples, note = args.note, n_part = args.n_part, func_list = func_list, combined_type = args.combined_type)
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_input', type=int, default=28*28)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='output/model.pth')
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--num_grids', type=int, default=8)
    parser.add_argument('--spline_order', type=int, default=3)
    parser.add_argument('--ds_name', type=str, default='mnist')
    parser.add_argument('--n_examples', type=int, default=0)
    parser.add_argument('--note', type=str, default='full')
    parser.add_argument('--n_part', type=float, default=0)
    parser.add_argument('--func_list', type=str, default='dog,rbf')
    parser.add_argument('--combined_type', type=str, default='quadratic')
    
    args = parser.parse_args()
    
    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)
    
#python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "mnist" --func_list "rbf,bs" --combined_type "quadratic"

#python run.py --mode "train" --model_name "efficient_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "fast_kan" --epochs 25 --batch_size 64 --n_input 3072 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "faster_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "mlp" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full"
