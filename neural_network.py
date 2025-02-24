import pandas as pandas
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import *
import matplotlib.pyplot as plt
import random


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
    

def get_data_loaders(train_months, val_months):
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


    #CHANGE CHANGE CHANGE PANIC ERROR
    all_csvs = train_csvs + val_csvs
    random.shuffle(all_csvs)

    train_csvs = all_csvs[:20000]
    val_csvs = all_csvs[20000:]
    

    X_train, y_train, pct_train = load_dataset(train_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval)



    X_val, y_val, pct_val = load_dataset(val_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval)

    train_percentage = np.mean(y_train) * 100
    val_percentage = np.mean(y_val) * 100

    print(f"Percentage of ones in y_train: {train_percentage:.2f}%")
    print(f"Percentage of ones in y_val: {val_percentage:.2f}%")

    train_dataset = Data(X_train, y_train)
    val_dataset = Data(X_val, y_val)


    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size = 32)  
    return train_data_loader, val_data_loader

def train_test_accuracy(model, train_data_loader, val_data_loader, epoch):
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
    print(f'Accuracy at Epoch {epoch + 1}: {val_accuracy} (validate)')
    print(f'Validation predictions: Ones = {ones_pred_val}, Zeros = {zeros_pred_val}')
    print(f'P(true label=1 | predicted label=1) (validate): {cond_prob_val}')

    return cond_prob_train, cond_prob_val

def graph_train_valid_error(train_accuracy, valid_accuracy):
    x_values = [i + 1 for i in range(len(train_accuracy))]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(x_values, valid_accuracy, label='Validation Accuracy', marker='x')
    plt.xlabel('Epoch (Index + 1)')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.show()



def train_model(train_months, val_months, num_epochs = 10):
    model = DenseTwoLayerNetwork()
    loss_function = nn.BCELoss()
    #common optimizer that changes learning rate on each individual parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) #0.001

    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months)

    train_acc, valid_acc = [], []



    for epoch in range(num_epochs):
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
        
    graph_train_valid_error(train_acc, valid_acc)


    
def main():
    train = ['jan2', 'jan1', 'dec1', 'dec2', 'nov1']
    val = ['feb1']
    train_model(train, val, num_epochs=200)


main()
