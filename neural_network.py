import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import *
import matplotlib.pyplot as plt
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu






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
        self.fully_connected_2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        hidden_layer = self.relu(self.fully_connected_1(x))
        logits = self.fully_connected_2(hidden_layer)
        return logits
    




def get_data_loaders(train_months, val_months, feature_names, randomized = False, silent = False, include_date = False):
    
    feature_names_static = []
    X_train, x_stat, y_train, X_val, x_stat, y_val, csvs = get_deep_learning_data(train_months, val_months, feature_names, feature_names_static, randomized = False, silent = False)

    train_dataset = Data(X_train, y_train)
    val_dataset = Data(X_val, y_val)

    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # change back
    val_data_loader = DataLoader(val_dataset, batch_size = 1)  #change back
    return train_data_loader, val_data_loader


def train_model(train_months, val_months, feature_names, num_epochs = 10, silent = False, randomized = False, include_date = False, one_weight = 1):

    
    feature_size = FIRST_N_MINUTES * len(feature_names)
    if include_date:
        feature_size += 2

    model = DenseTwoLayerNetwork(input_size = feature_size).to(device)
    pos_weight = torch.tensor([one_weight]).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #common optimizer that changes learning rate on each individual parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) #0.001

    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months, feature_names, randomized, silent = silent, include_date= include_date)

    train_acc, valid_acc = [], []



    for epoch in range(num_epochs):
        if epoch % 1 == 0 and not silent:
            t_acc, v_acc = train_test_accuracy(model, train_data_loader, val_data_loader, epoch)
            train_acc.append(t_acc)
            valid_acc.append(v_acc)
        model.train() #put model in traning mode
        for batch_X, stat_x, batch_y in train_data_loader:
            #gradients accumulate in pytorch, so this resets them all
            optimizer.zero_grad()

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X) #optimized better than forward function
            loss = loss_function(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()  #updates parameters

    if not silent:  
        graph_train_valid_error(train_acc, valid_acc, feature_names)
        train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent = False)
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

    train = ['sep', 'oct', 'nov', 'dec', 'jan']
    val = ['feb']

    folder_names_dict = get_folder_names(prefix = "data/", suffix = "_json_padded")
    

    train_folders = []
    for month in train:
        train_folders.append(folder_names_dict[month])

    val_folders = []
    for month in val:
        val_folders.append(folder_names_dict[month])
 
    #features = ['open','high','low','close','volume','usd_vol','price_change','vol_change','norm_open','mov_in_min']

    feature_names = ['volume', 'open', 'high', 'low', 'close']
    print(f'Running for {feature_names}')
    

    feature_combos = [['volume', 'vol_change', 'norm_open', 'mov_in_min'], ['volume', 'usd_vol', 'price_change', 'norm_open']]
    #feature_combos = [['volume', 'usd_vol', 'mov_in_min'], ['volume', 'price_change', 'norm_open'], ['vol_change', 'mov_in_min', 'norm_open']]
    #feature_combos = [['vol_change', 'norm_open'], ['vol_change', 'mov_in_min'], ['vol_change', 'usd_vol'], ['vol_change', 'price_change']]
    #feature_combos = [['volume', 'norm_open'], ['volume', 'mov_in_min'], ['volume', 'usd_vol'], ['volume', 'vol_change']]
    #feature_combos = [['volume'], ['usd_vol'], ['vol_change']]


    #grid_search(train, val, feature_combos, num_epochs= 200, times_run = 5)

    train_model(train, val, feature_names, num_epochs=100, randomized=False, include_date=False)


main()
