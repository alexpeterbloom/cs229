import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from util import *
import math

from lstm import TimeSeriesData, LSTM, get_data_loaders, train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu




def ensemble_train_test_accuracy(models, train_data_loader, val_data_loader, epoch, silent=False, csv_names=True):
    for model in models:
        model.eval()  # Put models in evaluation mode

    # --- Training Phase ---
    # Lists to store labels and predictions for the confusion matrix
    train_true_labels = []
    train_pred_labels = []
    
    ones_pred_train, zeros_pred_train = 0, 0
    train_total, train_correct = 0, 0
    true_positive_train = 0

    with torch.no_grad():  # Saves computation time
        for train_X_mov, train_X_stat, train_y in train_data_loader:
            train_X_mov, train_y = train_X_mov.to(device), train_y.to(device)
            if train_X_stat.numel() > 0:
                train_X_stat = train_X_stat.to(device)

            predictions = ensemble_predictions(train_X_mov, train_X_stat, models)
            binary_pred = (predictions > 0.5).squeeze(1)
            
            # Accumulate predictions and labels for later confusion matrix computation
            train_true_labels.append(train_y.cpu())
            train_pred_labels.append(binary_pred.cpu())

            train_total += train_y.size(0)  # Number of datapoints
            train_correct += (binary_pred == train_y.byte()).sum().item()
            ones_pred_train += binary_pred.sum().item()
            zeros_pred_train += (binary_pred == 0).sum().item()
            true_positive_train += ((binary_pred == 1) & (train_y.byte() == 1)).sum().item()

    train_accuracy = train_correct / train_total
    cond_prob_train = true_positive_train / ones_pred_train if ones_pred_train > 0 else 0

    # Concatenate all batches
    train_true_labels = torch.cat(train_true_labels)
    train_pred_labels = torch.cat(train_pred_labels)

    # Compute confusion matrix for training data
    TP_train = ((train_pred_labels == 1) & (train_true_labels.byte() == 1)).sum().item()
    TN_train = ((train_pred_labels == 0) & (train_true_labels.byte() == 0)).sum().item()
    FP_train = ((train_pred_labels == 1) & (train_true_labels.byte() == 0)).sum().item()
    FN_train = ((train_pred_labels == 0) & (train_true_labels.byte() == 1)).sum().item()
    confusion_train = [[TP_train, FP_train],
                         [FN_train, TN_train]]

    if not silent:
        print(f'Accuracy at Epoch {epoch + 1}: {train_accuracy:.4f} (train)')
        print(f'Training predictions: Ones = {ones_pred_train}, Zeros = {zeros_pred_train}')
        print(f'P(true label=1 | predicted label=1) (train): {cond_prob_train:.4f}')
        print("Confusion Matrix (Train):")
        print("                Predicted 1   Predicted 0")
        print(f"True 1:         {TP_train:<14} {FN_train}")
        print(f"True 0:         {FP_train:<14} {TN_train}")

    # --- Validation Phase ---
    val_true_labels = []
    val_pred_labels = []
    ones_pred_val, zeros_pred_val = 0, 0
    val_total, val_correct = 0, 0
    true_positive_val = 0

    with torch.no_grad():
        for val_X_mov, val_X_stat, val_y, _ in val_data_loader:
            val_X_mov, val_y = val_X_mov.to(device), val_y.to(device)
            if val_X_stat.numel() > 0:
                val_X_stat = val_X_stat.to(device)

            predictions = ensemble_predictions(val_X_mov, val_X_stat, models)
            binary_pred = (predictions > 0.5).squeeze(1)
            
            val_true_labels.append(val_y.cpu())
            val_pred_labels.append(binary_pred.cpu())

            val_total += val_y.size(0)
            val_correct += (binary_pred == val_y.byte()).sum().item()
            ones_pred_val += binary_pred.sum().item()
            zeros_pred_val += (binary_pred == 0).sum().item()
            true_positive_val += ((binary_pred == 1) & (val_y.byte() == 1)).sum().item()

    val_accuracy = val_correct / val_total
    cond_prob_val = true_positive_val / ones_pred_val if ones_pred_val > 0 else 0

    val_true_labels = torch.cat(val_true_labels)
    val_pred_labels = torch.cat(val_pred_labels)

    TP_val = ((val_pred_labels == 1) & (val_true_labels.byte() == 1)).sum().item()
    TN_val = ((val_pred_labels == 0) & (val_true_labels.byte() == 0)).sum().item()
    FP_val = ((val_pred_labels == 1) & (val_true_labels.byte() == 0)).sum().item()
    FN_val = ((val_pred_labels == 0) & (val_true_labels.byte() == 1)).sum().item()
    confusion_val = [[TP_val, FP_val],
                     [FN_val, TN_val]]

    if not silent:
        print(f'Accuracy at Epoch {epoch + 1}: {val_accuracy:.4f} (validate)')
        print(f'Validation predictions: Ones = {ones_pred_val}, Zeros = {zeros_pred_val}')
        print(f'P(true label=1 | predicted label=1) (validate): {cond_prob_val:.4f}')
        print("Confusion Matrix (Validate):")
        print("                Predicted 1   Predicted 0")
        print(f"True 1:         {TP_val:<14} {FN_val}")
        print(f"True 0:         {FP_val:<14} {TN_val}")

    return cond_prob_train, cond_prob_val, confusion_train, confusion_val


def train_ensemble(train_months, val_months, feature_names_mov, feature_names_static, params1, params2, params3, params4, params5, randomized = False, silent = True, batch_size_print = 30, graph = False):
    print("Getting data loader")
    train_data_loader, val_data_loader = get_data_loaders(train_months, val_months, feature_names_mov, feature_names_static, randomized, silent, batch_size= batch_size_print)
    print("BEGIN TRAINING MODEL 1")
    model1 = train_model(train_months, val_months, params1['num_epochs'], params1['hidden_size'], feature_names_mov, feature_names_static, randomized = randomized, silent = silent, batch_size = params1['batch_size'], dropout = params1['dropout'], lr = params1['lr'], train_data_loader= train_data_loader, val_data_loader= val_data_loader, graph = False)
    print("BEGIN TRAINING MODEL 2")
    model2 = train_model(train_months, val_months, params2['num_epochs'], params2['hidden_size'], feature_names_mov, feature_names_static, randomized = randomized, silent = silent, batch_size = params2['batch_size'], dropout = params2['dropout'], lr = params2['lr'], train_data_loader= train_data_loader, val_data_loader= val_data_loader, graph = False)
    print("BEGIN TRAINING MODEL 3")
    model3 = train_model(train_months, val_months, params3['num_epochs'], params3['hidden_size'], feature_names_mov, feature_names_static, randomized = randomized, silent = silent, batch_size = params3['batch_size'], dropout = params3['dropout'], lr = params3['lr'], train_data_loader=train_data_loader, val_data_loader=val_data_loader, graph = False)
    print("4")
    model4 = train_model(train_months, val_months, params4['num_epochs'], params4['hidden_size'], feature_names_mov, feature_names_static, randomized = randomized, silent = silent, batch_size = params4['batch_size'], dropout = params4['dropout'], lr = params4['lr'], train_data_loader=train_data_loader, val_data_loader=val_data_loader, graph = False)
    print("5")
    model5 = train_model(train_months, val_months, params5['num_epochs'], params5['hidden_size'], feature_names_mov, feature_names_static, randomized = randomized, silent = silent, batch_size = params5['batch_size'], dropout = params5['dropout'], lr = params5['lr'], train_data_loader=train_data_loader, val_data_loader=val_data_loader, graph = False)

    
    print("PRINTING RESULTS")
    ensemble_train_test_accuracy([model1, model2, model3, model4, model5], train_data_loader, val_data_loader, 100, silent = False, csv_names = True)

    print_ones_in_val(None, val_data_loader, ensemble = True, models = [model1, model2, model3, model4, model5])





if __name__ == '__main__':
    train = ['sep', 'oct', 'nov', 'dec', 'jan']
    val = ['feb']

    features_mov = ['open', 'volume', 'high', 'low']

    features_stat = [["btc_market_data", "vol_24hr"], ["btc_market_data", "price_change_24h"], ["btc_market_data", "price_change_7d"],
                     ["sol_market_data", "vol_24hr"], ["sol_market_data", "price_change_24h"], ["sol_market_data", "price_change_7d"],
                    ["creator_data", "other_coins_30_return"]]

    feature_stat= []
    params1, params2, params3, params4, params5 = {}, {}, {}, {}, {}
    
    params1['num_epochs'] = 25
    params1['hidden_size'] = 20
    params1['dropout'] = 0
    params1['batch_size'] = 1
    params1['lr'] = 0.001

    params2['num_epochs'] = 20
    params2['hidden_size'] = 20
    params2['dropout'] = 0
    params2['batch_size'] = 5
    params2['lr'] = 0.001

    params3['num_epochs'] = 30
    params3['hidden_size'] = 20
    params3['dropout'] = 0
    params3['batch_size'] = 10
    params3['lr'] = 0.001

    params4['num_epochs'] = 20
    params4['hidden_size'] = 15
    params4['dropout'] = 0
    params4['batch_size'] = 1
    params4['lr'] = 0.001

    params5['num_epochs'] = 20
    params5['hidden_size'] = 20
    params5['dropout'] = 0
    params5['batch_size'] = 1
    params5['lr'] = 0.001

    train_ensemble(train, val, features_mov, features_stat, params1, params2, params3, params4, params5, randomized = False, silent = False, batch_size_print = 30, graph = False)
 
