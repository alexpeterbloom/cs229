import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import datetime
import random
from torch.utils.data import Dataset, DataLoader
import torch
import json

VOTES_NEEDED = 4
CHANGE_NEEDED = 0.8
FIRST_N_MINUTES = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu


def read_our_json_file(file_name):
    with open(file_name, 'r') as f:
        json_data = json.load(f)

    df = pd.DataFrame(json_data["ohlcv_data"])
    return df, json_data


def print_ones_in_val(model, val_data_loader):
    model.eval()
    print("Printing all datapoints we predicted 1 for")
    total_return = 0
    total_capped_return = 0
    one_count = 0
    with torch.no_grad():
        for batch_x_mov, stat_x_mov, _, batch_file_names in val_data_loader:
            batch_x_mov = batch_x_mov.to(device)
            
            if stat_x_mov.numel() > 0:
                stat_x_mov = stat_x_mov.to(device)
            outputs = model(batch_x_mov, static_x = stat_x_mov)

            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).int().cpu().numpy().flatten() #cpu technically not necessary

            for i, pred in enumerate(predictions):
                if pred == 1:
                    #print(f'Predicted 1 for file: {batch_file_names[i]}')
                    change, capped_change = get_percent_return(batch_file_names[i])
                    one_count += 1
                    total_return += change
                    total_capped_return += capped_change
    print(f"Average capped change of {total_capped_return/one_count}")
    print(f"Average change of {total_return/one_count}")
        


    


def get_percent_return(csv_name):
    movement_df, json_data = read_our_json_file(csv_name)

    open_x = movement_df.iloc[FIRST_N_MINUTES, 1]

    #switched to this rather than final close for clarity
    final_open = movement_df.iloc[-1, 1]

    pct_change = ((final_open - open_x)/ open_x) * 100
    capped_pct_change = min(pct_change, 500)
    if pct_change > 1000:
        print(csv_name, 'big')
    return pct_change, capped_pct_change


def train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent = False, csv_names = True):
    model.eval() #put it in evaluation mode

    ones_pred_train, zeros_pred_train = 0, 0
    train_total, train_correct = 0, 0
    true_positive_train = 0
    with torch.no_grad(): #saves computation time
        for train_X_mov, train_X_stat, train_y in train_data_loader:
            train_X_mov, train_y = train_X_mov.to(device),  train_y.to(device)
            if train_X_stat.numel() > 0:
                train_X_stat = train_X_stat.to(device)
            predictions = torch.sigmoid(model(train_X_mov, static_x = train_X_stat))

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
        for val_X_mov, val_X_stat, val_y, _ in val_data_loader:
            val_X_mov, val_y = val_X_mov.to(device), val_y.to(device)
            if val_X_stat.numel() > 0:
                 val_X_stat = val_X_stat.to(device)
            predictions = torch.sigmoid(model(val_X_mov, static_x = val_X_stat))
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

def graph_train_valid_error(train_accuracy, valid_accuracy, feature_names_all):
    x_values = [i + 1 for i in range(len(train_accuracy))]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(x_values, valid_accuracy, label='Validation Accuracy', marker='x')
    plt.xlabel('Epoch (Index + 1)')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy over {feature_names_all.append(CHANGE_NEEDED)}')
    plt.legend()
    plt.show()



def get_deep_learning_data(train_months, val_months, feature_names_mov, feature_names_static, randomized = False, silent = False):
    #import training here

    folder_names_dict = get_folder_names(prefix = "data/", suffix = "_json_padded")
    

    train_folders = []
    for month in train_months:
        train_folders.append(folder_names_dict[month])

    val_folders = []
    for month in val_months:
        val_folders.append(folder_names_dict[month])


    print('train', train_folders)
    train_csvs = gather_all_csv(train_folders)
    val_csvs = gather_all_csv(val_folders)



    dif_from_begin_train, dif_from_begin_val, days_of_week_train, days_of_week_val = None, None, None, None


    if randomized:
            all_csvs = train_csvs + val_csvs
            random.shuffle(all_csvs)
            train_csvs = all_csvs[:20000]
            val_csvs = all_csvs[20000:]
        

    X_train_mov, X_train_stat, y_train, pct_train = load_dataset(train_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval, feature_names_mov, feature_names_static,
                                        silent = silent, days_of_week=days_of_week_train, days_from_begin=dif_from_begin_train)


    X_val_mov, X_val_stat, y_val, pct_val = load_dataset(val_csvs, CHANGE_NEEDED, FIRST_N_MINUTES, logistic_eval, feature_names_mov, feature_names_static,
                                        silent = silent, days_of_week=days_of_week_val, days_from_begin=dif_from_begin_val)


    train_percentage = np.mean(y_train) * 100
    val_percentage = np.mean(y_val) * 100

    if not silent:
        print(f"Percentage of ones in y_train: {train_percentage:.2f}%")
        print(f"Percentage of ones in y_val: {val_percentage:.2f}%")
        print(f'Average change of train: {sum(pct_train)/len(pct_train)}')
        print(f'Average change of test: {sum(pct_val)/len(pct_val)}')

    return X_train_mov, X_train_stat, y_train, X_val_mov, X_val_stat, y_val, val_csvs





def confirm_no_overlap(train_folders, test_folders):
    for train in train_folders:
        if train in test_folders:
            raise Exception("PANIC TRAINING AND TESTING OVERLAPPING")



def get_folder_names(prefix = "init_data/", suffix = "_json_padded"):
    all_batches = {}

    all_batches['sep'] = prefix + 'sep' + suffix
    all_batches['oct'] = prefix + 'oct' + suffix
    all_batches['nov'] = prefix + 'nov' + suffix
    all_batches['dec'] = prefix + 'dec' + suffix
    all_batches['jan'] = prefix + 'jan' + suffix
    all_batches['feb'] = prefix + 'feb' + suffix
    return all_batches


def get_static_features(json_data, static_feature_names):
    data = []
    for name in static_feature_names:
        sub_json = json_data
        for sub_category in name:
            sub_json = sub_json[sub_category]
        data.append(sub_json)

    if json_data['processed_data']['above_bar_15_min_in'] == 0 and json_data['processed_data']['above_bar_30_min_in'] == 0:
        raise ValueError("Datapoint should never have been in train or test set, died less than 15/30 minutes in")
    return data

def get_first_x_features_movement(df, x, feature_names, days_from_begin = None, day_of_week = None):
    #sub_df = df.iloc[:x, 1:6] #taking columns one to five

    sub_df = df.loc[:x-1, feature_names]
    flattened = sub_df.values.flatten()

    additional_features = []
    if days_from_begin is not None:
        additional_features.append(days_from_begin)
    if day_of_week is not None:
        additional_features.append(day_of_week)
    
    if additional_features:
        flattened = np.concatenate((flattened, np.array(additional_features)))


    return flattened





def continuous_eval(final_open, open_x, change):
    return ((final_open - open_x)/ open_x) * 100


def logistic_eval(final_open, open_x, change):
    label = 0 if final_open <= change * open_x else 1
    return label



def load_dataset(csv_files, change, x, evaluation_func, movement_feature_names, static_feature_names, store_full_df = False, silent = False, 
                days_of_week = None, days_from_begin = None, first_min = 30):
                
    X_mov, y, X_stat, pct_changes = [], [], [], []
    num_processed = 0
    full_dfs = []
    for i, csv_file in enumerate(csv_files):
        num_processed += 1

        if not os.path.isfile(csv_file) or os.path.getsize(csv_file) == 0:
            print(f'Problem: Failed to Read {csv_file}')
            continue
        
        
        try:
            df, json_data = read_our_json_file(csv_file)

            if df.empty:
                print(f'Problem: Encountered Empty Df in Util')
                continue

        except Exception as e:
            print(f'Problem: Failed To Read Dataframe in Util with error {e}')
            continue


        if json_data['processed_data']['above_bar_' + str(first_min) + "_min_in"] == 0:
            continue


        
        days_dist = None
        if days_from_begin != None:
            days_dist = days_from_begin[i]
        day_of_week = None
        if days_of_week != None:
            day_of_week = days_of_week[i]


        features_stat = get_static_features(json_data, static_feature_names)
        X_stat.append(features_stat)

        features_mov = get_first_x_features_movement(df, x, movement_feature_names, days_dist, day_of_week)
        if features_mov is None or df.shape[0] <= x:
            print("Problem: Not Enough Rows in Util")
            print("Problem 1")
            continue

        open_x = df.iloc[x, 1]
        if open_x <= 0:
            continue

        #switched to this rather than final close for clarity
        final_open = df.iloc[-1, 1]

        label = evaluation_func(final_open, open_x, change)

        pct_change = ((final_open - open_x)/ open_x) * 100
        pct_change = min(pct_change, 500)

        X_mov.append(features_mov)
        y.append(label)
        pct_changes.append(pct_change)

        if store_full_df:
            full_dfs.append(df)

    X_mov = np.array(X_mov)

    X_stat = np.array(X_stat)
    if len(static_feature_names) == 0:
        X_stat = None

    y = np.array(y)
    pct_changes = np.array(pct_changes)

    if not silent:
        print(f"Total CSV files processed: {num_processed}")
        print(f"Total in dataset: {len(X_mov)}")
    if store_full_df:
        return X_mov, X_stat, y, pct_changes, full_dfs
    
    return X_mov, X_stat, y, pct_changes





def graphTimeSeries(test_dfs, preds, x):
    all_time_series = []

    for i, df in enumerate(test_dfs):
        if preds[i] == 1: 
            open_x = df.iloc[x, 1]
            price_series = df.iloc[x:, 1].values
            movement_series = ((price_series - open_x) / open_x) * 100
            all_time_series.append(movement_series)

    all_time_series = np.array(all_time_series)

    if len(all_time_series) == 0:
        print("We didn't predict any coins")
    else:
        avg_time_series = all_time_series.mean(axis=0)

        plt.figure(figsize=(10,6))
        plt.plot(avg_time_series, label='Average movement for Predicted=1')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Minute Index (starting from 30)")
        plt.ylabel("Percent Movement from Minute 30 Open")
        plt.title("Average Time‐Series Movement (Predicted=1 Coins)")
        plt.legend()
        plt.grid(True)
        plt.show()

        final_avg_movement = avg_time_series[-1]
        print(f"Final average movement after 10 hours (predicted=1): {final_avg_movement:.2f}%")


def average_percent_change(pct_changes):
    return float(np.mean(pct_changes))


def graph_hist(data):
    plt.hist(data, bins=5, edgecolor='black')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Histogram Example')



def logisticVoting(all_preds):
    ones = all_preds.count(1)
    if ones >= VOTES_NEEDED:
        return 1
    return 0


def continuousVoting(all_preds):
    pred = sum(all_preds)/len(all_preds)
    return 1 if pred > 30 else 0


def combine_logistic_continuous_preds(logistic_predictions, continuous_predictions, voting_method = "Both_Confirm", threshold = 30, logistic_weight = 60):
    if voting_method == 'Both_Confirm':
        if len(logistic_predictions[0]) > 0:
            logistic_predicted = [1 if logisticVoting(all_preds) == 1 else 0 for all_preds in logistic_predictions]
            print(f'Logistic predicted {logistic_predicted.count(1)}')
        else:
            raise ValueError("No Logistic Models: Can't have Logistic and Continuous Vote")

        if len(continuous_predictions[0]) > 0: 
            continuous_predicted = [1 if continuousVoting(all_preds) == 1 else 0 for all_preds in continuous_predictions]
            print(f'Continuous predicted {continuous_predicted.count(1)}')
        else:
            raise ValueError("No Continuous Models: Can't Have Logistic and Continuous Vote")
        preds= []

        for l, c in zip(logistic_predicted, continuous_predicted):
            if l and c:
                preds.append(1)
            else:
                preds.append(0)

    elif voting_method == 'Total_Sum':

        preds = []
        total_datapoints = len(logistic_predictions)
        total_pred_each_datapoint = len(logistic_predictions[0]) + len(continuous_predictions[0])
        for i in range(total_datapoints):

            total_sum = 0
            log = logistic_predictions[i]
            for l in log:
                if l == 0:
                    total_sum -= logistic_weight
  
                else:
                    total_sum += logistic_weight


            cont = continuous_predictions[i]
            for c in cont:
                total_sum += c



            if total_sum/total_pred_each_datapoint > threshold:
                preds.append(1)
            else:
                preds.append(0)

    else:
        raise ValueError("Invalid Voting Method Inputted")
    
    return np.array(preds)
        

def quiet_eval(preds, y_test, pct_test, continuous = False):
    if continuous:
        new_y_test = [1 if i > 0 else 0 for i in y_test]
        new_preds = [1 if i > 0 else 0 for i in preds]
        preds, y_test = new_preds, new_y_test


    pred_1_changes = [p for p, pr in zip(pct_test, preds) if pr == 1]
    avg_1 = np.mean(pred_1_changes)
    num_ones_pred = np.sum(preds == 1)
    return avg_1, num_ones_pred
    

def evaluate(preds, y_test, pct_test, continuous = False):
    print()
    print()
    print()

    if continuous:
        new_y_test = [1 if i > 0 else 0 for i in y_test]
        new_preds = [1 if i > 0 else 0 for i in preds]
        preds, y_test = new_preds, new_y_test


    preds = np.array(preds)
    cm = confusion_matrix(y_test, preds)
    if cm.shape == (1, 1):
        print("Test set was all predicted to be same thing")
        print(cm)
        return
    
    tn, fp, fn, tp = cm.ravel()
    if (tn + fn == 0 or tp + fp == 0):
        print('We predicted all to be one type so we cannot evaluate')
        return
    

    total_test = len(y_test)
    print(f'Total Test Set Size: {total_test}')
    print()


    num_zeros = np.sum(y_test == 0)
    num_ones = np.sum(y_test == 1)
    print(f'In the actual dataset, there are {num_zeros} that are 0 and {num_ones} that are 1')
    print()

    num_zeros_pred = np.sum(preds == 0)
    num_ones_pred = np.sum(preds == 1)

    print(f'We predicted {num_zeros_pred} to be 0 and {num_ones_pred} to be 1')


    neg_pred_acc = (tn)/(tn + fn)
    pos_pred_acc = (tp)/(tp + fp)
    print(f'{round(100 * neg_pred_acc)}% of our 0 preds were correct')
    print(f'{round(100 * pos_pred_acc)}% of our 1 preds were correct')
    

    print('Confusion Matrix')
    print(cm)





    pred_0_changes = [p for p, pr in zip(pct_test, preds) if pr == 0]
    pred_1_changes = [p for p, pr in zip(pct_test, preds) if pr == 1]

 
    avg_0 = np.mean(pred_0_changes)
    avg_1 = np.mean(pred_1_changes)

    print(f'Average change conditional on predicting 0: {round(avg_0, 1)}')
    print(f'Average change conditional on predicting 1: {round(avg_1, 1)}')




#useful for both continuous and categorical
def train_model_and_pred(X_train, y_train, X_test, model):
    if len(X_train) == 0:
        print("No Training Examples")
        return
    if len(X_test) == 0:
        print("No testing samples")
        return
    
    if len(np.unique(y_train) ) < 2:
        print("Only one outcome in training data, can't train")
        return


    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


        

def randomize_with_date(train_csvs, val_csvs, days_of_week_train, dif_from_begin_train, days_of_week_val, dif_from_begin_val):
    train_data = list(zip(train_csvs, days_of_week_train, dif_from_begin_train))
    val_data = list(zip(val_csvs, days_of_week_val, dif_from_begin_val))
    combined_data = train_data + val_data

    random.shuffle(combined_data)
    train_data = combined_data[:20000]
    val_data = combined_data[20000:]


    train_csvs, days_of_week_train, dif_from_begin_train = zip(*train_data)
    train_csvs = list(train_csvs)
    days_of_week_train = list(days_of_week_train)
    dif_from_begin_train = list(dif_from_begin_train)


    val_csvs, days_of_week_val, dif_from_begin_val = zip(*val_data)
    val_csvs = list(val_csvs)
    days_of_week_val = list(days_of_week_val)
    dif_from_begin_val = list(dif_from_begin_val)
    return train_csvs, val_csvs, days_of_week_train, dif_from_begin_train, days_of_week_val, dif_from_begin_val

def gather_all_csv(folder_names):
    all_csv = []
    for folder in folder_names:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".json"):
                    all_csv.append(os.path.join(folder, fname))
    return all_csv



def clear_screen():
    for i in range(300):
        print()



    



