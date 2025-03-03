import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu


class TimeSeriesData(Dataset):
    def __init__(self, X, y, seq_length, num_features):
        self.y = y.astype(np.float32)
        X = X.astype(np.float32)
        self.X = X.reshape(-1, seq_length, num_features)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    

    

class LSTM(nn.Module):
    #num features is amount of datapoints we give for each minute
    def __init__(self, num_features = 5, hidden_size = 30, num_layers = 1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #"batch_first = True" added so input tensor dimensions work
        self.lstm_block = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

        self.fc_final = nn.Linear(hidden_size, 1)

    def forward(self, x): #x is batch
        #fine to initialize h0 and c0 with zeros because they're not parameters and therefore
        #we don't have symmetry problem. h0, c0 are tensors all data in batch.

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #".to(x.device)" moves data to gpu we run on. Test with/without it and seemed to run quicker

        lstm_out, _ = self.lstm_block(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :] #we only care about hidden layer at final time step

        final_out = self.fc_final(lstm_out)
        return final_out
    

def get_data_loaders(train_months, val_months, feature_names, randomized = False, silent = False, include_date = False):
    

    X_train, y_train, X_val, y_val = get_deep_learning_data(train_months, val_months, feature_names, randomized, silent, include_date)

    num_features = len(feature_names)


    train_dataset = TimeSeriesData(X_train, y_train, seq_length=FIRST_N_MINUTES, num_features = num_features)
    val_dataset = TimeSeriesData(X_val, y_val, seq_length=FIRST_N_MINUTES, num_features = num_features)

    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=50, shuffle = True) 
    val_data_loader = DataLoader(val_dataset, batch_size = 50)  
    return train_data_loader, val_data_loader


def train_model(train_months, val_months, num_epochs, hidden_size, feature_names, randomized = False, silent = False, include_date = False):
    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months, feature_names, randomized, silent, include_date)
    
    input_size = len(feature_names)
    num_layers = 1
    model = LSTM(input_size, hidden_size, num_layers)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model.to(device)

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

        if epoch % 5 == 0:
            train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent)
 
    
if __name__ == '__main__':
    train = ['jan2', 'jan1', 'dec1', 'dec2', 'nov1']
    val = ['feb1']

    features = ['norm_open', 'volume', 'mov_in_min']
    print(f'Running for {features}')
    train_model(train, val, num_epochs = 100, hidden_size = 20, feature_names = features, randomized = False, silent = False, include_date = False)









