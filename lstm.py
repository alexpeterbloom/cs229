import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu


class TimeSeriesData(Dataset):
    def __init__(self, X, y, seq_length, num_features, file_names = None):
        self.y = y.astype(np.float32)
        X = X.astype(np.float32)
        self.X = X.reshape(-1, seq_length, num_features)
        self.file_names = file_names

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        x =  torch.tensor(self.X[idx]) 
        y = torch.tensor(self.y[idx])
        if self.file_names is not None:
            return x, y, self.file_names[idx]
        else:
            return x, y
    
    

    

class LSTM(nn.Module):
    #num features is amount of datapoints we give for each minute
    def __init__(self, num_features = 5, hidden_size = 30, num_layers = 1, dropout_rate = 0.15):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #"batch_first = True" added so input tensor dimensions work
        self.lstm_block = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_final = nn.Linear(hidden_size, 1)

    def forward(self, x): #x is batch
        #fine to initialize h0 and c0 with zeros because they're not parameters and therefore
        #we don't have symmetry problem. h0, c0 are tensors all data in batch.

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #".to(x.device)" moves data to gpu we run on. Test with/without it and seemed to run quicker

        lstm_out, _ = self.lstm_block(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :] #we only care about hidden layer at final time step
        lstm_out = self.dropout(lstm_out)
        final_out = self.fc_final(lstm_out)
        return final_out
    

def get_data_loaders(train_months, val_months, feature_names, randomized = False, silent = False, include_date = False, batch_size = 1):
    

    X_train, y_train, X_val, y_val, file_names_val = get_deep_learning_data(train_months, val_months, feature_names, randomized, silent, include_date)

    num_features = len(feature_names)


    train_dataset = TimeSeriesData(X_train, y_train, seq_length=FIRST_N_MINUTES, num_features = num_features)
    val_dataset = TimeSeriesData(X_val, y_val, seq_length=FIRST_N_MINUTES, num_features = num_features, file_names = file_names_val)

    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True) 
    val_data_loader = DataLoader(val_dataset, batch_size = batch_size)  
    return train_data_loader, val_data_loader


def train_model(train_months, val_months, num_epochs, hidden_size, feature_names, randomized = False, silent = False, include_date = False, batch_size = 30, dropout = 0):
    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months, feature_names, randomized, silent, include_date, batch_size= batch_size)
    
    input_size = len(feature_names)
    num_layers = 1
    model = LSTM(input_size, hidden_size, num_layers, dropout_rate= dropout)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model.to(device)
    train_test_accuracy(model, train_data_loader, val_data_loader, 0, silent)
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()  #in training mode

        for batch_x, batch_y in train_data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)
            if outputs.shape != batch_y.shape: #making sure dimensions line up
                batch_y = batch_y.unsqueeze(1)
            
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 3 == 0:
            t_acc, v_acc = train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent)
            train_acc.append(t_acc)
            val_acc.append(v_acc)
    print_ones_in_val(model, val_data_loader)
    graph_train_valid_error(train_acc, val_acc, feature_names + [dropout])



if __name__ == '__main__':
    train = [ 'sep1', 'sep2', 'oct1', 'oct2', 'nov1', 'nov2', 'dec1', 'dec2']
    val = ['jan1', 'jan2', 'feb1', 'feb2']


    features = ['volume', 'norm_open', 'mov_in_min']
    epochs = 25
    hidden = 20
    is_random = False
    batch = 10
    dropout = 0
    print(f'Running for {features}, {epochs}, {hidden}, {is_random}, {batch}, {dropout}')
    train_model(train, val, num_epochs = epochs, hidden_size = hidden, feature_names = features, randomized = is_random, silent = False, include_date = False, batch_size = batch, dropout = dropout)









