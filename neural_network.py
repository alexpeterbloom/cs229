import pandas as pandas
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import *


#what fraction of original value must it maintain to be counted as a 1
CHANGE_NEEDED = 1
FIRST_N_MINUTES = 30


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.y)


class DenseTwoLayerNetwork(nn.Module): #inheriting from PyTorch nn Module
    def __init__(self, input_size = 150, hidden_size = FIRST_N_MINUTES):
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
    

def get_data_loaders(train_months, val_months):
    #import training here

    folder_names_dict = get_folder_names()


    train_csvs = []
    for month in train_months:
        train_csvs += folder_names_dict[month]

    val_csvs = []
    for month in val_months:
        val_csvs += folder_names_dict[month]

    X_train, y_train, pct_train = load_dataset(train_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval)

    X_val, y_val, pct_val = load_dataset(val_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval)


    train_dataset = Data(X_train, y_train)
    val_dataset = Data(X_val, y_val)

    print(len(train_dataset))

    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size = 32)  
    return train_data_loader, val_data_loader


def train_model(train_months, val_months, num_epochs = 10):
    model = DenseTwoLayerNetwork()
    loss_function = nn.BCELoss()
    #common optimizer that changes learning rate on each individual parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months)

    for epoch in range(num_epochs):
        model.train() #put model in traning mode
        for batch_X, batch_y in train_data_loader:
            #gradients accumulate in pytorch, so this resets them all
            optimizer.zero_grad()
            outputs = model(batch_X) #optimized better than forward function
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()  #updates parameters
        
        model.eval() #put it in evaluation mode
        total, correct = 0, 0
        with torch.no_grad(): #saves computation time
            for val_X, val_y in val_data_loader:
                predictions = model(val_X)
                binary_pred = predictions > 0.5
                total += val_y.size(0) #number of datapoints
                correct += (binary_pred == val_y.byte()).sum().item()

        accuracy = correct/total
        print(f'Validation Accuracy at Epoch {epoch + 1}: {accuracy}')


    
def main():
    train = ['nov1', 'dec1', 'dec2', 'jan1']
    val = ['jan2']
    train_model(train, val)


main()
