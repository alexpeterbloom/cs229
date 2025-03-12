import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu


class TimeSeriesData(Dataset):
    def __init__(self, X, y, seq_length, num_features, static = None, file_names = None):
        self.y = y.astype(np.float32)
        X = X.astype(np.float32)
        self.X = X.reshape(-1, seq_length, num_features)
        if static is not None and static.size != 0:
            self.static = static.astype(np.float32)
        else:
            self.static = None
        self.file_names = file_names

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x =  torch.tensor(self.X[idx]) 
        y = torch.tensor(self.y[idx])
        if self.static is not None:
            static_x = torch.tensor(self.static[idx])
            if self.file_names is not None:
                return x, static_x, y, self.file_names[idx]
            else:
                return x, static_x, y
        else:
            dummy_static = torch.empty(0)
            if self.file_names is not None:
                return x, dummy_static, y, self.file_names[idx]
            else:
                return x, dummy_static, y
    
    

    

class LSTM(nn.Module):
    #num features is amount of datapoints we give for each minute
    def __init__(self, num_features = 5, hidden_size = 30, num_layers = 1, static_input_size = None, static_hidden_size = 16, dropout_rate = 0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #"batch_first = True" added so input tensor dimensions work
        self.lstm_block = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        if static_input_size is not None:
            self.static_fc = nn.Linear(static_input_size, static_hidden_size)
            self.fc_final = nn.Linear(hidden_size + static_hidden_size, 1)
        else:
            self.static_fc = None
            self.fc_final = nn.Linear(hidden_size, 1)

    def forward(self, x, static_x = None): #x is batch
        #fine to initialize h0 and c0 with zeros because they're not parameters and therefore
        #we don't have symmetry problem. h0, c0 are tensors all data in batch.

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #".to(x.device)" moves data to gpu we run on. Test with/without it and seemed to run quicker

        lstm_out, _ = self.lstm_block(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :] #we only care about hidden layer at final time step
        lstm_out = self.dropout(lstm_out)

        if self.static_fc is not None:
            static_out = torch.relu(self.static_fc(static_x))
            combined = torch.cat((lstm_out, static_out), dim = 1)
            final_out = self.fc_final(combined)
        else:
            final_out = self.fc_final(lstm_out)

        return final_out
    

def get_data_loaders(train_months, val_months, feature_names_mov, feature_names_static, randomized = False, silent = False, batch_size = 1):
    

    X_train_mov, X_train_stat, y_train, X_val_mov, X_val_stat, y_val, file_names_val = get_deep_learning_data(train_months, val_months, feature_names_mov, feature_names_static, randomized, silent)

    num_features_mov = len(feature_names_mov)


    train_dataset = TimeSeriesData(X_train_mov, y_train, seq_length=FIRST_N_MINUTES, num_features = num_features_mov, static = X_train_stat)
    val_dataset = TimeSeriesData(X_val_mov, y_val, seq_length=FIRST_N_MINUTES, num_features = num_features_mov, static = X_val_stat, file_names = file_names_val)

    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True) 
    val_data_loader = DataLoader(val_dataset, batch_size = batch_size)  
    return train_data_loader, val_data_loader


def train_model(train_months, val_months, num_epochs, hidden_size, feature_names_mov, feature_names_static, randomized = False, silent = False, batch_size = 30, dropout = 0):
    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months, feature_names_mov, feature_names_static, randomized, silent, batch_size= batch_size)
    
    input_size_mov = len(feature_names_mov)

    input_size_static = len(feature_names_static)
    if input_size_static == 0:
        input_size_static = None

    num_layers = 1
    model = LSTM(input_size_mov, hidden_size, num_layers, static_input_size = input_size_static, dropout_rate= dropout)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model.to(device)
    train_test_accuracy(model, train_data_loader, val_data_loader, 0, silent)
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()  #in training mode

        for batch_x_mov, batch_x_stat, batch_y in train_data_loader:
            batch_x_mov = batch_x_mov.to(device)
            if batch_x_stat.numel() > 0:
                batch_x_stat = batch_x_stat.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x_mov, static_x = batch_x_stat)
            if outputs.shape != batch_y.shape: #making sure dimensions line up
                batch_y = batch_y.unsqueeze(1)
            
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            t_acc, v_acc = train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent)
            train_acc.append(t_acc)
            val_acc.append(v_acc)
    print_ones_in_val(model, val_data_loader)
    graph_train_valid_error(train_acc, val_acc, feature_names_mov + feature_names_static + [dropout])



if __name__ == '__main__':
    train = ['oct', 'nov', 'dec', 'jan']
    val = ['feb']


    features_mov = ['open', 'volume', 'high', 'low']
    features_stat = [['processed_data', 'creator_data', 'other_coins_30_return']]
    features_stat = []
    epochs = 25
    hidden = 20
    is_random = False
    batch = 1
    dropout = 0.2
    print(f'Running for {features_mov}, {features_stat}, {epochs}, {hidden}, {is_random}, {batch}, {dropout}, {CHANGE_NEEDED}')
    train_model(train, val, num_epochs = epochs, hidden_size = hidden, feature_names_mov = features_mov, feature_names_static=features_stat, randomized = is_random, silent = False, batch_size = batch, dropout = dropout)

