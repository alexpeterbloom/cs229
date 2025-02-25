import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import *
import matplotlib.pyplot as plt
import random
import numpy as np


#what fraction of original value must it maintain to be counted as a 1
CHANGE_NEEDED = 1
FIRST_N_MINUTES = 30


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DenseTwoLayerNetwork(nn.Module): #inheriting from PyTorch nn Module
    def __init__(self, input_size = 30, hidden_size = 30):
        #initializes funcationality of nn.Module
        super(DenseTwoLayerNetwork, self).__init__()
        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fully_connected_2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        hidden_layer = self.relu(self.fully_connected_1(x))
        preds = self.sigmoid(self.fully_connected_2(hidden_layer))
        return preds
    

def get_data_loaders(train_months, val_months, feature_names, randomized = False, silent = False, include_date = False):
    #import training here

    folder_names_dict = get_folder_names(suffix = "_padded_extra_features")


    train_folders = []
    for month in train_months:
        train_folders += folder_names_dict[month]

    val_folders = []
    for month in val_months:
        val_folders += folder_names_dict[month]

    train_csvs = gather_all_csv(train_folders)
    val_csvs = gather_all_csv(val_folders)

    dif_from_begin_train, dif_from_begin_val, days_of_week_train, days_of_week_val = None, None, None, None
    if include_date:
        days_of_week_train, dif_from_begin_train = gather_all_date_info(train_csvs)
        days_of_week_val, dif_from_begin_val = gather_all_date_info(val_csvs)


    if randomized:
        if include_date:
            train_csvs, val_csvs, days_of_week_train, dif_from_begin_train, days_of_week_val, dif_from_begin_val = \
            randomize_with_date(train_csvs, val_csvs, days_of_week_train, dif_from_begin_train, days_of_week_val, dif_from_begin_val)
        else:
            all_csvs = train_csvs + val_csvs
            random.shuffle(all_csvs)
            train_csvs = all_csvs[:20000]
            val_csvs = all_csvs[20000:]
        

    X_train, y_train, pct_train = load_dataset(train_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval, feature_names,
                                        silent = silent, days_of_week=days_of_week_train, days_from_begin=dif_from_begin_train)



    X_val, y_val, pct_val = load_dataset(val_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval, feature_names,
                                        silent = silent, days_of_week=days_of_week_val, days_from_begin=dif_from_begin_val)


    train_percentage = np.mean(y_train) * 100
    val_percentage = np.mean(y_val) * 100

    if not silent:
        print(f"Percentage of ones in y_train: {train_percentage:.2f}%")
        print(f"Percentage of ones in y_val: {val_percentage:.2f}%")

    train_dataset = Data(X_train, y_train)
    val_dataset = Data(X_val, y_val)


    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # change back
    val_data_loader = DataLoader(val_dataset, batch_size = 1)  #change back
    return train_data_loader, val_data_loader

def train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent = False):
    model.eval() #put it in evaluation mode

    ones_pred_train, zeros_pred_train = 0, 0
    train_total, train_correct = 0, 0
    true_positive_train = 0
    with torch.no_grad(): #saves computation time
        for train_X, train_y in train_data_loader:
            predictions = model(train_X)
            binary_pred = (predictions > 0.5).squeeze(1)
            train_total += train_y.size(0) #number of datapoints
            train_correct += (binary_pred == train_y.byte()).sum().item()
            ones_pred_train += binary_pred.sum().item() 
            zeros_pred_train += (binary_pred == 0).sum().item()
            true_positive_train += ((binary_pred == 1) & (train_y.byte() == 1)).sum().item()
    
    train_accuracy = train_correct/train_total
    cond_prob_train = true_positive_train / ones_pred_train if ones_pred_train > 0 else 0
    if not silent:
        print(f'Accuracy at Epoch {epoch + 1}: {train_accuracy} (train)')
        print(f'Training predictions: Ones = {ones_pred_train}, Zeros = {zeros_pred_train}')
        print(f'P(true label=1 | predicted label=1) (train): {cond_prob_train}')



    ones_pred_val, zeros_pred_val = 0, 0
    val_total, val_correct = 0, 0
    true_positive_val = 0
    with torch.no_grad():
        for val_X, val_y in val_data_loader:
            predictions = model(val_X)
            binary_pred = (predictions > 0.5).squeeze(1)
            val_total += val_y.size(0) 
            val_correct += (binary_pred == val_y.byte()).sum().item()
            ones_pred_val += binary_pred.sum().item()
            zeros_pred_val += (binary_pred == 0).sum().item()
            true_positive_val += ((binary_pred == 1) & (val_y.byte() == 1)).sum().item()

    val_accuracy = val_correct/val_total
    cond_prob_val = true_positive_val / ones_pred_val if ones_pred_val > 0 else 0
    if not silent:
        print(f'Accuracy at Epoch {epoch + 1}: {val_accuracy} (validate)')
        print(f'Validation predictions: Ones = {ones_pred_val}, Zeros = {zeros_pred_val}')
        print(f'P(true label=1 | predicted label=1) (validate): {cond_prob_val}')

    return cond_prob_train, cond_prob_val

def graph_train_valid_error(train_accuracy, valid_accuracy, feature_names):
    x_values = [i + 1 for i in range(len(train_accuracy))]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(x_values, valid_accuracy, label='Validation Accuracy', marker='x')
    plt.xlabel('Epoch (Index + 1)')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy over {feature_names}')
    plt.legend()
    plt.show()



def train_model(train_months, val_months, feature_names, num_epochs = 10, silent = False, randomized = False, include_date = False):
    feature_size = FIRST_N_MINUTES * len(feature_names)
    if include_date:
        feature_size += 2

    model = DenseTwoLayerNetwork(input_size = feature_size)
    loss_function = nn.BCELoss()
    #common optimizer that changes learning rate on each individual parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) #0.001

    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months, feature_names, randomized, silent = silent, include_date= include_date)

    train_acc, valid_acc = [], []



    for epoch in range(num_epochs):
        if epoch % 5 == 0 and not silent:
            t_acc, v_acc = train_test_accuracy(model, train_data_loader, val_data_loader, epoch)
            train_acc.append(t_acc)
            valid_acc.append(v_acc)
        model.train() #put model in traning mode
        for batch_X, batch_y in train_data_loader:
            #gradients accumulate in pytorch, so this resets them all
            optimizer.zero_grad()
            outputs = model(batch_X) #optimized better than forward function
            loss = loss_function(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()  #updates parameters

    if not silent:  
        graph_train_valid_error(train_acc, valid_acc, feature_names)
    else:
        t_acc, v_acc = train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent = True)
        return t_acc, v_acc


def grid_search(train_months, val_months, feature_combos, num_epochs, times_run):
    clear_screen()
    print(f'Running on {feature_combos}')
    print()
    for combo in feature_combos:
        print(f'Currently running on combo {combo}')
        total_test = 0
        total_val = 0
        for i in range(times_run):
            t_acc, v_acc = train_model(train_months, val_months, combo, num_epochs= num_epochs, silent = True, randomized= True)
            total_test += t_acc
            total_val += v_acc
        avg_test = total_test/times_run
        avg_val = total_val/times_run
        print(f'Average test: {avg_test}, Average val: {avg_val}')
        print()
        print()
    

        

    
def main():
    train = ['jan2', 'jan1', 'dec1', 'dec2', 'nov1']
    val = ['feb1']

    #features = ['open','high','low','close','volume','usd_vol','price_change','vol_change','norm_open','mov_in_min']

    feature_names = ['volume', 'vol_change', 'norm_open', 'mov_in_min']
    print(f'Running for {feature_names}')
    

    feature_combos = [['volume', 'vol_change', 'norm_open', 'mov_in_min'], ['volume', 'usd_vol', 'price_change', 'norm_open']]
    #feature_combos = [['volume', 'usd_vol', 'mov_in_min'], ['volume', 'price_change', 'norm_open'], ['vol_change', 'mov_in_min', 'norm_open']]
    #feature_combos = [['vol_change', 'norm_open'], ['vol_change', 'mov_in_min'], ['vol_change', 'usd_vol'], ['vol_change', 'price_change']]
    #feature_combos = [['volume', 'norm_open'], ['volume', 'mov_in_min'], ['volume', 'usd_vol'], ['volume', 'vol_change']]
    #feature_combos = [['volume'], ['usd_vol'], ['vol_change']]


    #grid_search(train, val, feature_combos, num_epochs= 200, times_run = 5)

    train_model(train, val, feature_names, num_epochs=200, randomized=True, include_date=False)


main()
