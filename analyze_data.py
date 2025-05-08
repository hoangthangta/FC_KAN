from file_io import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt  
import math

import os


def get_best_of(ds_name = 'sl_mnist', ds_rate = 'full', show_result = True):

    #models = ['bsrbf_kan', 'bsrbf_kan_gan', 'fast_kan', 'fast_kan_gan', 'faster_kan', 'faster_kan_gan', 'efficient_kan', 'efficient_kan_gan', 'mlp',  'mlp_gan']
    models = ['fast_kan', 'brd_kan']
    
    result_dict = {}
    for model in models:
    
        #print('model: ', model)
        best_acc = 0
        
        best_item = {}
        for i in range(0, 5):
            #data = read_list_from_jsonl('output/' + str(i) + '/' + model + '_' + ds_name + '.json')
            file_name = 'output/' + ds_name + '/' + model + '/'
            
            if ('gan' in model and 'mlp_gan' not in model):
                file_name += 'kan_gan__' + model.replace('_gan', '') + '__' +  ds_name +'__' + ds_rate + '_' + str(i) + '.json'
            elif ('mlp_gan' in model):
                file_name += 'mlp_gan__' +  ds_name +'__' + ds_rate + '_' + str(i) + '.json'
            else:
                file_name += model + '__' +  ds_name +'__' + ds_rate + '_' + str(i) + '.json'
            
            #print('file_name: ', file_name)
            data = read_list_from_jsonl(file_name)
            
            best_temp = {}
            for item in data:
                if (item['epoch'] == data[-2]['best_epoch']):
                    best_temp = item
                    break
            
            if not best_item:
                best_item = best_temp
                best_item['training_time'] = data[-1]['training time']
            else:
                if (best_temp['val_accuracy'] > best_item['val_accuracy']):
                    best_item = best_temp
                    best_item['training_time'] = data[-1]['training time']
        
        result_dict[model] = best_item
        #print('--', best_item)
        '''{'epoch': 14, 'val_accuracy': 0.9763136942675159, 'train_accuracy': 1.0, 'f1_macro': 0.975968880719942, 'pre_macro': 0.9761056495229932, 're_macro': 0.9758770695720186, 'best_epoch': 14, 'val_loss': 0.08504310913434845, 'train_loss': 0.002305890594693849, 'training_time': 222.60320043563843}'''
    
    if (show_result == False): result_dict
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Training time (seconds)', '']
    print('get_best_of: ', ds_name, ds_rate)
    print(' | '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' | '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        lst = ['', k, round(v['train_accuracy']*100, 2), round(v['val_accuracy']*100, 2), round(v['f1_macro']*100, 2), round(v['pre_macro']*100, 2), round(v['re_macro']*100, 2), int(v['training_time']), '']
        print(' | '.join(str(x) for x in lst))
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict
    
def show_plot(ds_rate = 'full'):

    # create subplots
    fig, axs = plt.subplots(1, 2, figsize=(8,5))

    # create data
    models = ['bsrbf_kan', 'fast_kan', 'faster_kan', 'efficient_kan', 'mlp',  'mfc_kan']
    
    ds_name = 'mnist'
    field1 = 'train_loss'
    field2 = 'train_d_loss'
    data_list = []
    for model in models:
        
        if (model == 'mfc_kan'):
            
            url = 'output/' + ds_name + '/' + model + '/' + model + '__' + ds_name + '__dog-bs__quadratic__' + ds_rate + '_0.json'
        else:
            url = 'output/' + ds_name + '/' + model + '/' + model + '__' + ds_name + '__' + ds_rate + '_0.json'
        data = read_list_from_jsonl(url)[:-1]
        data_list.append(data)
    
    # First plot
    epochs = range(1, len(data_list[0]) + 1)
    #marker_list = ['o', 's', '^', '>', '<', '*']
    #line_list = ['-', '--', '-.', ':', 'dashed', 'solid']
    #color_list = ['blue', 'red', 'green', 'orange', 'deeppink', 'brown']
    
    marker_list = ['o', 's', '^', '>', '<', '*', 'D', 'p', 'H', 'X']
    line_list = ['-', '--', '-.', ':', 'None', 'solid', 'dashed', 'dashdot', 'dotted', '--']
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'deeppink', 'magenta', 'brown', 'gray', 'deeppink']
    
    for epoch, marker, color, model, line in zip(epochs, marker_list, color_list, models, line_list):  
        if (model == 'mfc_kan'):
            model = 'fc_kan'
        try:
            subdata = [math.log(x[field1]) for x in data_list[epoch-1]]
        except:
            try:
                subdata = [math.log(x[field2]) for x in data_list[epoch-1]]
            except:
                pass
        axs[0].plot(epochs, subdata, marker=marker, linestyle='-.', label=model, color=color)
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('log(' + field1 + ')')
    axs[0].set_title('MNIST')
    axs[0].legend()
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    axs[0].grid(True)

   
    ds_name = 'fashion_mnist'
    
    field1 = 'train_loss'
    field2 = 'train_d_loss'
    data_list = []
    for model in models:
        
        if (model == 'mfc_kan'):
            
            url = 'output/' + ds_name + '/' + model + '/' + model + '__' + ds_name + '__dog-bs__quadratic__' + ds_rate + '_0.json'
        else:
            url = 'output/' + ds_name + '/' + model + '/' + model + '__' + ds_name + '__' + ds_rate + '_0.json'
        data = read_list_from_jsonl(url)[:-1]
        data_list.append(data)

    # First plot
    epochs = range(1, len(data_list[0]) + 1)
    #marker_list = ['o', 's', '^', '>', '<', '*']
    #line_list = ['-', '--', '-.', ':', 'dashed', 'solid']
    #color_list = ['blue', 'red', 'green', 'orange', 'deeppink', 'brown']
    
    marker_list = ['o', 's', '^', '>', '<', '*', 'D', 'p', 'H', 'X']
    line_list = ['-', '--', '-.', ':', 'None', 'solid', 'dashed', 'dashdot', 'dotted', '--']
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'deeppink', 'magenta', 'brown', 'gray', 'deeppink']
    
    for epoch, marker, color, model, line in zip(epochs, marker_list, color_list, models, line_list):   
        if (model == 'mfc_kan'):
            model = 'fc_kan'
        try:
            subdata = [math.log(x[field1]) for x in data_list[epoch-1]]
        except:
            try:
                subdata = [math.log(x[field2]) for x in data_list[epoch-1]]
            except:
                pass
        axs[1].plot(epochs, subdata, marker=marker, linestyle='-.', label=model, color=color)
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('log(' + field1 + ')')
    axs[1].set_title('Fashion-MNIST')
    axs[1].legend()
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    axs[1].grid(True)

    plt.tight_layout()
    plt.show()  

def show_average_plot(ds_name = 'mnist', field1 = 'train_loss', field2 = 'train_d_loss'):

    # create data
    models = ['bsrbf_kan', 'bsrbf_kan_gan', 'fast_kan', 'fast_kan_gan', 'faster_kan', 'faster_kan_gan', 'efficient_kan', 'efficient_kan_gan', 'mlp',  'mlp_gan']
    
    pcts = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 'full']

    # create subplots
    #fig, axs = plt.subplots(1, 2, figsize=(8,5))
    plt.figure(figsize=(8, 5))
    
    # get data
    data_list = []
    for x in pcts:
        data_list.append(get_average(ds_name = ds_name, ds_rate = str(x), show_result = False))
    
    # reorganize data
    matrix = []
    for idx, item in enumerate(data_list):
        value_list = []
        for k, v in item.items():
            try:
                value_list.append(v[field1])
            except:
                try:
                    value_list.append(v[field2])
                except:
                    pass
        matrix.append(value_list)
    
    # Transpose the matrix
    matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    print(matrix)

    # First plot
    index_list = range(1, len(matrix) + 1)
    marker_list = ['o', 's', '^', '>', '<', '*', 'D', 'p', 'H', 'X']
    line_list = ['-', '--', '-.', ':', 'None', 'solid', 'dashed', 'dashdot', 'dotted', '--']
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'deeppink']
    
    for idx, marker, color, model, line in zip(index_list, marker_list, color_list, models, line_list):
        #if (idx % 2 == 1): continue
        subdata = [x for x in matrix[idx-1]]
        print(subdata)
        plt.plot(pcts, subdata, marker=marker, linestyle='--', label=model, color=color)
    plt.xlabel('data rate')
    plt.ylabel(field1)
    plt.title(ds_name.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  
        
def get_average(ds_name = 'fashion_mnist', ds_rate = 'full', show_result = True):

    #models = ['bsrbf_kan', 'fast_kan', 'faster_kan', 'efficient_kan', 'mlp']
    models = ['bsrbf_kan', 'fast_kan', 'faster_kan', 'efficient_kan']

    result_dict = {}
    for model in models:
    
        #print('model: ', model)

        item_list = []
        time_list = []
        for i in range(0, 5):
            # output\fashion_mnist\bsrbf_kan_gan\kan_gan__bsrbf_kan__fashion_mnist__full_4.json

            file_name = 'output/' + ds_name + '/' + model + '/'
            
            
            if ('gan' in model and 'mlp_gan' not in model):
                file_name += 'kan_gan__' + model.replace('_gan', '') + '__' +  ds_name +'__' + ds_rate + '_' + str(i) + '.json'
            elif ('mlp_gan' in model):
                file_name += 'mlp_gan__' +  ds_name +'__' + ds_rate + '_' + str(i) + '.json'
            else:
                file_name += model + '__' +  ds_name +'__' + ds_rate + '_' + str(i) + '.json'
            
            #print('--', file_name)
            
            #print('file_name: ', file_name)
            data = read_list_from_jsonl(file_name)
            
            best_temp = {}
            for item in data:
                if (item['epoch'] == data[-2]['best_epoch']):
                    best_temp = item
                    break
            item_list.append(best_temp)
            time_list.append(data[-1]['training time'])
        
        # {"epoch": 15, "val_accuracy": 0.9746218152866242, "train_accuracy": 1.0, "f1_macro": 0.9742424560731686, "pre_macro": 0.9743872748374693, "re_macro": 0.9741579526141277, "best_epoch": 15, "val_loss": 0.08656448658387195, "train_loss": 0.001877448516985665}
        
        
        avg_item = {}
        rate_list = ['val_accuracy', 'train_accuracy', 'f1_macro', 're_macro', 'pre_macro']
        for k, v in item_list[0].items():

            if (k in rate_list):
                avg_item[k] = sum([x[k]*100 for x in item_list])/len(item_list)
                mean = np.mean([x[k]*100 for x in item_list])
                std_dev = np.std([x[k]*100 for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
            else:
                avg_item[k] = sum([x[k] for x in item_list])/len(item_list)
                mean = np.mean([x[k] for x in item_list])
                std_dev = np.std([x[k] for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
            

            # Calculate the standard error (uncertainty)
            standard_error = std_dev / np.sqrt(len([x[k]*100 for x in item_list]))
            avg_item['mean_uncertainty' + '_' + k] = f"{mean:.2f} ± {standard_error:.2f}"
        
        avg_item['training_time'] = sum(time_list)/len(time_list)
        
        avg_item['training_time'] = f"{avg_item['training_time']:.2f}"
            
        #print('--', avg_item)
        result_dict[model] = avg_item
        
    if (show_result == False): result_dict
    
    
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Training time (seconds)', '']
    print('get_average: ', ds_name, ds_rate)
    print(' | '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' & '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        #print(v)
        lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_f1_macro'], v['mean_uncertainty_pre_macro'], v['mean_uncertainty_re_macro'], v['training_time'], '']
        print(' & '.join(str(x) for x in lst))
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict

def get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'quadratic', show_result = True):

    models = ['fc_kan']
    func_list = ['bs-dog']
    #func_list = ['dog-bs', 'dog-rbf', 'dog-base', 'bs-rbf', 'bs-base', 'rbf-base']
    
    result_dict = {}
    for model in models:

        for func in func_list:
            item_list = []
            time_list = []
            for i in range(0, 5):

                file_name = 'output/' + ds_name + '/' + model + '/'
                file_name += model + '__' + ds_name + '__' + func + '__' +  combined_type +  '__' + ds_rate + '_' + str(i) + '.json'
                
                
                print('file_name: ', file_name)
                data = read_list_from_jsonl(file_name, limit = 0)
                
                best_temp = {}
                for item in data:
                    if (item['epoch'] == data[-2]['best_epoch']):
                        best_temp = item
                        break
                
                if (len(data) == 0): continue
                item_list.append(best_temp)
                time_list.append(data[-1]['training time'])
            
            
            avg_item = {}
            rate_list = ['val_accuracy', 'train_accuracy', 'f1_macro', 're_macro', 'pre_macro']
            for k, v in item_list[0].items():

                if (k in rate_list):
                    avg_item[k] = sum([x[k]*100 for x in item_list])/len(item_list)
                    mean = np.mean([x[k]*100 for x in item_list])
                    std_dev = np.std([x[k]*100 for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                else:
                    avg_item[k] = sum([x[k] for x in item_list])/len(item_list)
                    mean = np.mean([x[k] for x in item_list])
                    std_dev = np.std([x[k] for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                
                # Calculate the standard error (uncertainty)
                standard_error = std_dev / np.sqrt(len([x[k]*100 for x in item_list]))
                avg_item['mean_uncertainty' + '_' + k] = f"{mean:.2f} ± {standard_error:.2f}"
            
            avg_item['training_time'] = sum(time_list)/len(time_list)
                
            #print('--', avg_item)
            result_dict[model +'_' + func + '_' + combined_type] = avg_item
        
    if (show_result == False): result_dict
    
    
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Training time (seconds)', '']
    print('get_average: ', ds_name, ds_rate)
    print(' & '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' & '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        #print(v)
        lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_f1_macro'], round(v['training_time'],2), '']
        print(' & '.join(str(x) for x in lst))
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict
    

def get_ab(ds_name = 'fashion_mnist', folder = 'ab_fashion'):

    models = ['full', 'no_base', 'no_base_no_layer_norm', 'no_bs', 'no_bs_no_rbf', 'no_layer_norm', 'no_rbf']
    
    i = 0
    full_lst = []
    for model in models:
    
        #print('model: ', model)

        data = read_list_from_jsonl('output/' +  folder + '/' + model + '/bsrbf_kan_' + ds_name + '.json')
            
        best_temp = {}
        for item in data:
            if (item['epoch'] == data[-2]['best_epoch']):
                best_temp = item
                break
        
        if (i == 0):
            full_lst = [model, round(best_temp['train_accuracy']*100, 3), round(best_temp['val_accuracy']*100,3), round(best_temp['f1_macro']*100,3)]
        
        
        lst = [model, round(best_temp['train_accuracy']*100, 3), round(best_temp['val_accuracy']*100,3), round(best_temp['f1_macro']*100,3)]
        
        if (i > 0):
            temp_lst = [lst[0]]
            for x, y in zip(lst[1:], full_lst[1:]):
                temp_lst.append(round(x - y, 3))
            lst = temp_lst
            
        print(' & '.join(str(x) for x in lst))
        i += 1


def show_ab_plot():

    # create subplots
    fig, axs = plt.subplots(1, 2, figsize=(8,5))
    
    ds_name = 'mnist'
    folder = 'ab_mnist'
    # create data
    methods = ['full', 'no_base', 'no_base_no_layer_norm', 'no_bs', 'no_bs_no_rbf', 'no_layer_norm', 'no_rbf']
    
    field = 'train_loss'
    data_list = []
    for method in methods:
        data = read_list_from_jsonl('output/' +  folder + '/' + method + '/bsrbf_kan_' + ds_name + '.json')[:-1]
        data_list.append(data)

    # First plot
    epochs = range(1, len(data_list[0]) + 1)
    marker_list = ['o', 's', '^', '>', '<', '*', 's']
    line_list = ['-', '--', '-.', ':', 'dashed', 'solid', '-:']
    color_list = ['blue', 'red', 'green', 'orange', 'deeppink', 'brown', 'purple']
    
    for epoch, marker, color, method, line in zip(epochs, marker_list, color_list, methods, line_list):
        
        subdata = [math.log(x[field]) for x in data_list[epoch-1]]
        axs[0].plot(epochs, subdata, marker=marker, linestyle='-.', label=method, color=color)
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('log(' + field + ')')
    axs[0].set_title('MNIST')
    axs[0].legend()
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    axs[0].grid(True)

    # First plot
    '''field = 'val_loss'
    epochs = range(1, len(data_list[0]) + 1)
    marker_list = ['o', 's', '^', '>', '<', '*', 'x', 'D']
    line_list = ['-', '--', '-.', ':', 'dashed', 'solid', 'dashdot', 'dotted']
    color_list = ['blue', 'red', 'green', 'orange', 'deeppink', 'brown', 'olive', 'purple']
    
    for epoch, marker, color, method, line in zip(epochs, marker_list, color_list, methods, line_list):   
        subdata = [math.log(x[field])*100000000 for x in data_list[epoch-1]]
        axs[0, 1].plot(epochs, subdata, marker=marker, linestyle='-.', label=method, color=color)
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('log(' + field + ')')
    axs[0, 1].set_title('MNIST')
    axs[0, 1].legend()
    #axs[1].set_xticks([])
    #axs[1].set_yticks([])
    axs[0, 1].grid(True)'''
    
    #------
    ds_name = 'fashion_mnist'
    folder = 'ab_fashion_mnist'
    
    field = 'train_loss'
    data_list = []
    for method in methods:
        data = read_list_from_jsonl('output/' +  folder + '/' + method + '/bsrbf_kan_' + ds_name + '.json')[:-1]
        data_list.append(data)

    # First plot
    epochs = range(1, len(data_list[0]) + 1)
    marker_list = ['o', 's', '^', '>', '<', '*', 's']
    line_list = ['-', '--', '-.', ':', 'dashed', 'solid', '-:']
    color_list = ['blue', 'red', 'green', 'orange', 'deeppink', 'brown', 'purple']
    
    for epoch, marker, color, method, line in zip(epochs, marker_list, color_list, methods, line_list):   
        subdata = [math.log(x[field]) for x in data_list[epoch-1]]
        axs[1].plot(epochs, subdata, marker=marker, linestyle='-.', label=method, color=color)
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('log(' + field + ')')
    axs[1].set_title('FashionMNIST')
    axs[1].legend()
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    axs[1].grid(True)

    '''# First plot
    field = 'val_loss'
    epochs = range(1, len(data_list[0]) + 1)
    marker_list = ['o', 's', '^', '>', '<', '*', 'x', 'D']
    line_list = ['-', '--', '-.', ':', 'dashed', 'solid', 'dashdot', 'dotted']
    color_list = ['blue', 'red', 'green', 'orange', 'deeppink', 'brown', 'olive', 'purple']
    
    for epoch, marker, color, method, line in zip(epochs, marker_list, color_list, methods, line_list):   
        subdata = [math.log(x[field])*100000000 for x in data_list[epoch-1]]
        axs[1, 1].plot(epochs, subdata, marker=marker, linestyle='-.', label=method, color=color)
    axs[1, 1].set_xlabel('epochs')
    axs[1, 1].set_ylabel('log(' + field + ')')
    axs[1, 1].set_title('FashionMNIST')
    axs[1, 1].legend()
    #axs[1].set_xticks([])
    #axs[1].set_yticks([])
    axs[1, 1].grid(True)'''

    plt.tight_layout()
    plt.show()          


def get_average_single_prkan(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'none', show_result = True):

    models = ['prkan']
    func_list = ['rbf']
    norm_types = ['batch', 'layer']
    #norm_types = ['none']
    activation = 'silu'
    #methods = ['conv1d_1', 'base', 'ds', 'fw', 'attention', 'conv1d_2']
    methods = ['conv2d']
    
    
    result_dict = {}
    for model in models:
        for norm_type in norm_types:
            for func in func_list:
                for method in methods:
                    item_list = []
                    time_list = []
                    for i in range(0, 5):
                        #file_name = 'papers/PRKAN/norm_pos_1_conv2d/' + ds_name + '/' + model + '/'
                        file_name = 'output/' + ds_name + '/' + model + '/'
                        file_name += model + '__' + ds_name + '__' + func + '__' +  norm_type +  '__' + activation +  '__'  + method +  '__' + combined_type + '__'  + ds_rate + '_' + str(i) + '.json'
                        data = read_list_from_jsonl(file_name, limit = 0)
                        best_temp = {}
                        for item in data:
                            if (item['epoch'] == data[-2]['best_epoch']):
                                best_temp = item
                                break
                        if (len(data) == 0): continue
                        item_list.append(best_temp)
                        time_list.append(data[-1]['training time'])

                    avg_item = {}
                    rate_list = ['val_accuracy', 'train_accuracy', 'f1_macro', 're_macro', 'pre_macro']
                    for k, v in item_list[0].items():
                        if (k in rate_list):
                            avg_item[k] = sum([x[k]*100 for x in item_list])/len(item_list)
                            mean = np.mean([x[k]*100 for x in item_list])
                            std_dev = np.std([x[k]*100 for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                        else:
                            avg_item[k] = sum([x[k] for x in item_list])/len(item_list)
                            mean = np.mean([x[k] for x in item_list])
                            std_dev = np.std([x[k] for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                        
                        # Calculate the standard error (uncertainty)
                        standard_error = std_dev / np.sqrt(len([x[k]*100 for x in item_list]))
                        avg_item['mean_uncertainty' + '_' + k] = f"{mean:.2f} ± {standard_error:.2f}"
                    avg_item['training_time'] = sum(time_list)/len(time_list)
                    result_dict[model + '_' + func + '_' + norm_type + '_' + activation + '_' + method + '_none'] = avg_item
                    
    if (show_result == False):
        print(len(result_dict))
    
    print(len(result_dict))
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Training time (seconds)', '']
    print('get_average: ', ds_name, ds_rate)
    print(' & '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' & '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        #print(v)
        lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_f1_macro'], round(v['training_time'],2), '']
        print(' & '.join(str(x) for x in lst))
        #print('---------')
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict


def get_average_pr_relu_kan(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'none', show_result = True):

    models = ['af_kan']
    #norm_types = ['batch', 'layer', 'none']
    norm_types = ['layer']
    activations = ['silu']
    #activations = ['relu', 'elu', 'gelu', 'selu', 'leaky_relu', 'softplus', 'tanh', 'sigmoid']
    methods = ['global_attn']
    #methods = ['global_attn', 'spatial_attn', 'simple_linear']
    #func_list = ['quad1']
    func_list = ['quad1', 'quad2', 'sum', 'prod', 'sum_prod', 'cubic1', 'cubic2']
    
    result_dict = {}
    for model in models:
        for norm_type in norm_types:
            for method in methods:
                for activation in activations:
                    for func in func_list:
                        item_list = []
                        time_list = []
                        for i in range(0, 5):
                            #pr_relu_kan__mnist__batch__gelu__global_attn__none__full_0
                            file_name = 'output/' + ds_name + '/' + model + '/'
                            file_name += model + '__' + ds_name + '__' + norm_type +  '__' + activation +  '__'  + method +  '__' + combined_type + '__' + func + '__' + ds_rate + '_' + str(i) + '.json'
                            #print('file_name: ', file_name)
                            data = read_list_from_jsonl(file_name, limit = 0)
                            best_temp = {}
                            for item in data:
                                if (item['epoch'] == data[-2]['best_epoch']):
                                    best_temp = item
                                    break
                            if (len(data) == 0): continue
                            item_list.append(best_temp)
                            time_list.append(data[-1]['training time'])

                        avg_item = {}
                        rate_list = ['val_accuracy', 'train_accuracy', 'test_accuracy', 'val_f1_macro', 'val_re_macro', 'val_pre_macro', 'test_f1_macro', 'test_re_macro', 'test_pre_macro', 'f1_macro', 'pre_macro', 're_macro']
                        for k, v in item_list[0].items():
                            if (k in rate_list):
                                avg_item[k] = sum([x[k]*100 for x in item_list])/len(item_list)
                                mean = np.mean([x[k]*100 for x in item_list])
                                std_dev = np.std([x[k]*100 for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                            else:
                                avg_item[k] = sum([x[k] for x in item_list])/len(item_list)
                                mean = np.mean([x[k] for x in item_list])
                                std_dev = np.std([x[k] for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                            
                            # Calculate the standard error (uncertainty)
                            standard_error = std_dev / np.sqrt(len([x[k]*100 for x in item_list]))
                            avg_item['mean_uncertainty' + '_' + k] = f"{mean:.2f} ± {standard_error:.2f}"
                        avg_item['training_time'] = sum(time_list)/len(time_list)
                        result_dict[model + '_' + norm_type + '_' + activation + '_' + method + '_' + combined_type + '_' + func + '_' + ds_rate] = avg_item
                
    if (show_result == False):
        print(len(result_dict))
    
    print(len(result_dict))
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Val Macro F1', 'Test Accuracy', 'Test Macro F1', 'Training time (seconds)', '']
    print('get_average: ', ds_name, ds_rate)
    print(' & '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' & '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        
        if (ds_name == 'cal_si'):
            lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_val_f1_macro'], v['mean_uncertainty_test_accuracy'], v['mean_uncertainty_test_f1_macro'], round(v['training_time'],2), '']
            print(' & '.join(str(x) for x in lst))
            #print('---------')
        else:
            lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_f1_macro'], round(v['training_time'],2), '']
            print(' & '.join(str(x) for x in lst))
            #print('---------')
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict

def get_average_relu_kan(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'none', show_result = True):

    models = ['relu_kan']
    #norm_types = ['batch', 'layer', 'none']
    norm_types = ['layer']
    activations = ['relu']
    result_dict = {}
    for model in models:
        for norm_type in norm_types:
   
            for activation in activations:
                
                item_list = []
                time_list = []
                for i in range(0, 5):
                    #pr_relu_kan__mnist__batch__gelu__global_attn__none__full_0
                    file_name = 'output/' + ds_name + '/' + model + '/'
                    #file_name += model + '__' + ds_name + '__' + norm_type +  '__' + activation +  '__'  + method +  '__' + combined_type + '__' + func + '__' + ds_rate + '_' + str(i) + '.json'
                    file_name += model + '__' + ds_name + '__' + ds_rate + '_' + str(i) + '.json'
                    print('file_name: ', file_name)
                    data = read_list_from_jsonl(file_name, limit = 0)
                    best_temp = {}
                    for item in data:
                        if (item['epoch'] == data[-2]['best_epoch']):
                            best_temp = item
                            break
                    if (len(data) == 0): continue
                    item_list.append(best_temp)
                    time_list.append(data[-1]['training time'])

                avg_item = {}
                rate_list = ['val_accuracy', 'train_accuracy', 'test_accuracy', 'val_f1_macro', 'val_re_macro', 'val_pre_macro', 'test_f1_macro', 'test_re_macro', 'test_pre_macro', 'f1_macro', 'pre_macro', 're_macro']
                for k, v in item_list[0].items():
                    if (k in rate_list):
                        avg_item[k] = sum([x[k]*100 for x in item_list])/len(item_list)
                        mean = np.mean([x[k]*100 for x in item_list])
                        std_dev = np.std([x[k]*100 for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                    else:
                        avg_item[k] = sum([x[k] for x in item_list])/len(item_list)
                        mean = np.mean([x[k] for x in item_list])
                        std_dev = np.std([x[k] for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                    
                    # Calculate the standard error (uncertainty)
                    standard_error = std_dev / np.sqrt(len([x[k]*100 for x in item_list]))
                    avg_item['mean_uncertainty' + '_' + k] = f"{mean:.2f} ± {standard_error:.2f}"
                avg_item['training_time'] = sum(time_list)/len(time_list)
                #result_dict[model + '_' + norm_type + '_' + activation + '_' + method + '_' + combined_type + '_' + func + '_' + ds_rate] = avg_item
                result_dict[model + '_' + ds_rate] = avg_item
            
    if (show_result == False):
        print(len(result_dict))
    
    print(len(result_dict))
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Val Macro F1', 'Test Accuracy', 'Test Macro F1', 'Training time (seconds)', '']
    print('get_average: ', ds_name, ds_rate)
    print(' & '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' & '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        
        if (ds_name == 'cal_si'):
            lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_val_f1_macro'], v['mean_uncertainty_test_accuracy'], v['mean_uncertainty_test_f1_macro'], round(v['training_time'],2), '']
            print(' & '.join(str(x) for x in lst))
            #print('---------')
        else:
            lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_f1_macro'], round(v['training_time'],2), '']
            print(' & '.join(str(x) for x in lst))
            #print('---------')
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict


def get_result(path = 'output', ds_name = 'mnist', models = ['relu_kan'], show_result = True):

    result_dict = {}
    for model in models:
  
        file_dict = get_file_dict(path = path, ds_name = ds_name, model = model)
        
        for key, values in file_dict.items():
            item_list = []
            time_list = []
            
            # for each key
            for file_name in values:
                data = read_list_from_jsonl(file_name, limit = 0)
                best_temp = {}
                for item in data:
                    if (item['epoch'] == data[-2]['best_epoch']):
                        best_temp = item
                        break
                if (len(data) == 0): continue
                item_list.append(best_temp)
                time_list.append(data[-1]['training time'])
        
            if (len(item_list) == 0): 
                print("Error: [item_list] empty!")
                continue
            
            #print(item_list)
            #print('------')
            avg_item = {}
            rate_list = ['val_accuracy', 'train_accuracy', 'test_accuracy', 'val_f1_macro', 'val_re_macro', 'val_pre_macro', 'test_f1_macro', 'test_re_macro', 'test_pre_macro', 'f1_macro', 'pre_macro', 're_macro']
            for k, v in item_list[0].items():
                #print(k, v)
                #print('-----------')
                if (k in rate_list):
                    avg_item[k] = sum([x[k]*100 for x in item_list])/len(item_list)
                    #print(avg_item[k], k)
                    #print('--------------')
                    mean = np.mean([x[k]*100 for x in item_list])
                    std_dev = np.std([x[k]*100 for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
                else:
                    avg_item[k] = sum([x[k] for x in item_list])/len(item_list)
                    mean = np.mean([x[k] for x in item_list])
                    std_dev = np.std([x[k] for x in item_list], ddof=1)  # Using ddof=1 for sample standard deviation
            
                # Calculate the standard error (uncertainty)
                standard_error = std_dev / np.sqrt(len([x[k]*100 for x in item_list]))
                avg_item['mean_uncertainty' + '_' + k] = f"{mean:.2f} ± {standard_error:.2f}"
            avg_item['training_time'] = sum(time_list)/len(time_list)
            result_dict[key + '_avg_' + str(len(values))] = avg_item

    # show results --------------------------------
    if (show_result == False):
        print(len(result_dict))
    
    print(len(result_dict))
    lst = ['', 'Network', 'Training Accuracy', 'Val Accuracy', 'Val Macro F1', 'Test Accuracy', 'Test Macro F1', 'Training time (seconds)', '']
    print('get_average: ', ds_name)
    print(' & '.join(str(x) for x in lst))
    lst2 = [''] + ['-------------']*(len(lst)-2) + ['']
    print(' & '.join(str(x) for x in lst2))
    for k, v in result_dict.items():
        if (ds_name == 'cal_si'):
            lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_val_f1_macro'], v['mean_uncertainty_test_accuracy'], v['mean_uncertainty_test_f1_macro'], round(v['training_time'],2), '']
            print(' & '.join(str(x) for x in lst))
            #print('---------')
        else:
            lst = ['', k, v['mean_uncertainty_train_accuracy'], v['mean_uncertainty_val_accuracy'], v['mean_uncertainty_f1_macro'], round(v['training_time'],2), '']
            print(' & '.join(str(x) for x in lst))
            #print('---------')
    print('------------------------------------------------------------------------------------------------------------')
    return result_dict
    
def get_file_dict(path = 'output', ds_name = 'mnist', model = 'relu_kan'):
    
    path = 'output' + '/' + ds_name + '/' + model
    files = os.listdir(path)  
    file_dict = {}
    for file in files:
        extension = os.path.splitext(file)[1]  # Returns ".txt"
        file_name = os.path.splitext(file)[0]
        
        if (extension == '.json'):
            try:
                num = int(file_name[-1:])
            except Exception as e:
                print('Error "get_file_dict": ', e)
                return file_dict
            
            conf  = file_name[:-2]
            
            if (conf not in file_dict):
                file_dict[conf] = [path + '/' + file]
            else:
                file_dict[conf].append(path + '/' + file)
    print('file_dict: ', file_dict)
    return file_dict


    
if __name__ == "__main__":
    
    get_result(path = 'output', ds_name = 'mnist', models = ['cnn'], show_result = True)
    get_result(path = 'output', ds_name = 'fashion_mnist', models = ['cnn'], show_result = True)
    get_result(path = 'output', ds_name = 'cifar10', models = ['cnn'], show_result = True)
    
    #get_average_relu_kan(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'none', show_result = True)
    
    #get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'quadratic', show_result = True)
    
    #get_average_pr_relu_kan(ds_name = 'cal_si', ds_rate = 'full', combined_type = 'none', show_result = True)
    #get_average_pr_relu_kan(ds_name = 'mnist', ds_rate = 'full', combined_type = 'none', show_result = True)
    #get_average_pr_relu_kan(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'none', show_result = True)

    
    #get_best_of(ds_name = 'mnist', ds_rate = 'full', show_result = True)
    
    #get_average(ds_name = 'fashion_mnist', ds_rate = 'full', show_result = True)
    #get_average(ds_name = 'mnist', ds_rate = 'full', show_result = True)
    '''get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'cubic', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'concat_linear', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'max', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'min', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = 'full', combined_type = 'mean', show_result = True)'''
    
    '''get_average_fc(ds_name = 'mnist', ds_rate = 'full', combined_type = 'sum', show_result = True)
    get_average_fc(ds_name = 'mnist', ds_rate = 'full', combined_type = 'product', show_result = True)
    get_average_fc(ds_name = 'mnist', ds_rate = 'full', combined_type = 'sum_product', show_result = True)'''
    
    #get_average_fc(ds_name = 'mnist', ds_rate = 'full', combined_type = 'quadratic', show_result = True)
    
    #get_average_fc(ds_name = 'mnist', ds_rate = 'full', combined_type = 'quadratic2', show_result = True)
    
    #get_average(ds_name = 'fashion_mnist', ds_rate = 'full', show_result = True)
    #get_average_fc(ds_name = 'mnist', ds_rate = 'full', combined_type = 'quadratic', show_result = True)

    '''get_average_fc(ds_name = 'mnist', ds_rate = '0.01', show_result = True)
    get_average_fc(ds_name = 'mnist', ds_rate = '0.05', show_result = True)
    get_average_fc(ds_name = 'mnist', ds_rate = '0.1', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = '0.01', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = '0.05', show_result = True)
    get_average_fc(ds_name = 'fashion_mnist', ds_rate = '0.1', show_result = True)'''
    
    #show_plot(ds_rate = 'full')
    #show_average_plot(ds_name = 'sl_mnist', field1 = 'val_accuracy', field2 = 'val_accuracy')
    
    #get_ab(ds_name = 'fashion_mnist', folder = 'ab_fashion_mnist')
    #get_ab(ds_name = 'mnist', folder = 'ab_mnist')
    #show_ab_plot()
    
    #get_average_single_prkan(ds_name = 'mnist', ds_rate = 'full', show_result = True)
    #get_average_single_prkan(ds_name = 'fashion_mnist', ds_rate = 'full', show_result = True)
    