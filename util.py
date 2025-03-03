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

VOTES_NEEDED = 4
CHANGE_NEEDED = 1
FIRST_N_MINUTES = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using m2 gpu


def train_test_accuracy(model, train_data_loader, val_data_loader, epoch, silent = False):
    model.eval() #put it in evaluation mode

    ones_pred_train, zeros_pred_train = 0, 0
    train_total, train_correct = 0, 0
    true_positive_train = 0
    with torch.no_grad(): #saves computation time
        for train_X, train_y in train_data_loader:
            train_X, train_y = train_X.to(device), train_y.to(device)
            predictions = model(train_X)
            binary_pred = (predictions > 0).squeeze(1)
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
            val_X, val_y = val_X.to(device), val_y.to(device)
            predictions = model(val_X)
            binary_pred = (predictions > 0).squeeze(1)
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



def get_deep_learning_data(train_months, val_months, feature_names, randomized = False, silent = False, include_date = False):
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
        print('here')
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


    return X_train, y_train, X_val, y_val





def confirm_no_overlap(train_folders, test_folders):
    for train in train_folders:
        if train in test_folders:
            raise Exception("PANIC TRAINING AND TESTING OVERLAPPING")



def get_folder_names(prefix = "data/", suffix = "_ohlcv_padded_low_volume_dropped"):
    all_batches = {}

    feb_days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

    feb_data = [prefix + 'feb' + day + suffix for day in feb_days]
    all_batches['feb1'] = feb_data

    january_train1 = [prefix + "jan0" + str(i) + suffix for i in range(1, 10)]
    january_train2 = [prefix + "jan" + str(i) + suffix for i in range(10, 16)]
    january_half_one = january_train1 + january_train2
    all_batches['jan1'] = january_half_one


    january_test = [prefix + "jan" + str(i) + suffix for i in range(16, 32)]
    all_batches['jan2'] = january_test

    dec1 = [prefix + "dec0" + str(i) + suffix for i in range(1, 10)]
    dec2 = [prefix + "dec" + str(i) + suffix for i in range(10, 32)]

    all_batches['dec1'] = dec1
    all_batches['dec2'] = dec2
    

    nov = [prefix + "nov0" + str(i) + suffix for i in range(1, 10)]
    nov.append(prefix + "nov10" + suffix)
    all_batches['nov1'] = nov

    return all_batches




def get_first_x_features(df, x, feature_names, days_from_begin = None, day_of_week = None):
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



def load_dataset(csv_files, change, x, evaluation_func, feature_names, store_full_df = False, silent = False, 
                days_of_week = None, days_from_begin = None):
    X, y, pct_changes = [], [], []
    num_processed = 0
    full_dfs = []
    for i, csv_file in enumerate(csv_files):
        num_processed += 1

        if not os.path.isfile(csv_file) or os.path.getsize(csv_file) == 0:
            print(f'Problem: Failed to Read {csv_file}')
            continue
        
        
        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                print(f'Problem: Encountered Empty Df in Util')
                continue
        except Exception as e:
            print(f'Problem: Failed To Read Dataframe in Util with error {e}')
            continue
        
        days_dist = None
        if days_from_begin != None:
            days_dist = days_from_begin[i]
        day_of_week = None
        if days_of_week != None:
            day_of_week = days_of_week[i]

        features = get_first_x_features(df, x, feature_names, days_dist, day_of_week)
        
        if features is None or df.shape[0] <= x:
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



        X.append(features)
        y.append(label)
        pct_changes.append(pct_change)

        if store_full_df:
            full_dfs.append(df)

    X = np.array(X)
    y = np.array(y)
    pct_changes = np.array(pct_changes)

    if not silent:
        print(f"Total CSV files processed: {num_processed}")
        print(f"Total in dataset: {len(X)}")
    if store_full_df:
        return X, y, pct_changes, full_dfs
    
    return X, y, pct_changes





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
        print('here')
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

            sum = 0
            log = logistic_predictions[i]
            for l in log:
                if l == 0:
                    sum -= logistic_weight
  
                else:
                    sum += logistic_weight


            cont = continuous_predictions[i]
            for c in cont:
                sum += c



            if sum/total_pred_each_datapoint > threshold:
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


def gather_all_date_info(folder_names):
    month_map = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }

    day_map = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }

    days_of_week, dif_from_begin_list = [], []
    for folder in folder_names:
        info = folder.split("/")[1]
        date = info.split("_")[0]
        month = date[:3]
        day = int(date[3:])
        year = 2025
        if month == 'nov' or month == 'dec':
            year = 2024
        
        month = month_map[month]
        input_date = datetime.date(int(year), int(month), int(day))
        
        target_date = datetime.date(2024, 11, 1)
        
        day_of_week = input_date.strftime("%A")
        day_of_week = day_map[day_of_week]
        days_since_beginning = (input_date - target_date).days
        days_of_week.append(day_of_week)
        dif_from_begin_list.append(days_since_beginning)
    return days_of_week, dif_from_begin_list
        

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
    all_days = []
    for folder in folder_names:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    all_csv.append(os.path.join(folder, fname))
    return all_csv

def clear_screen():
    for i in range(300):
        print()



    



