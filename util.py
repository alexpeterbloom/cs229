import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time



VOTES_NEEDED = 4


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




def get_first_x_features(df, x):
    #sub_df = df.iloc[:x, 1:6] #taking columns one to five
    col_names = ['price_change']
    
    sub_df = df.loc[:x-1, col_names]
    flattened = sub_df.values.flatten()
    return flattened





def continuous_eval(final_open, open_x, change):
    return ((final_open - open_x)/ open_x) * 100


def logistic_eval(final_open, open_x, change):
    label = 0 if final_open <= change * open_x else 1
    return label


def load_only_x_points(csv_files, x):
    X = []
    num_processed = 0
    for csv_file in csv_files:
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

        features = get_first_x_features(df, x)

        if features is None or df.shape[0] < x:
            print("Problem: Not Enough Rows in Util")
            print()
            time.sleep(50)
            continue
    

        X.append(features)
    X = np.array(X)

    print(f"Total CSV files processed in testing Data with no Y: {num_processed}")
    print(f"Total in dataset: {len(X)}")

    return X



def load_dataset(csv_files, change, x, evaluation_func, store_full_df = False, silent = False):
    X, y, pct_changes = [], [], []
    num_processed = 0
    full_dfs = []
    for csv_file in csv_files:
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
        

        features = get_first_x_features(df, x)
        
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



    



