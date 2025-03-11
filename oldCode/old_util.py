import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import math
from datetime import datetime
import time

#util: contains utility functions used in many files


#def get_day_of_the_week(month, day):
#    upper_month = month.capitalize()
#    months_last_year = ['Nov', 'Dec']
#    if upper_month in months_last_year:
#        current_year = 2024
#    else:
#        current_year = 2025
#    date_str = f"{upper_month} {day} {current_year}"
#    date_obj = datetime.strptime(date_str, "%b %d %Y")
#    return date_obj.weekday()

def get_day_of_the_week(month, day):
    return 1



#optimize: change which features you're taking
#optimize: change how many minutes in the beginning you're taking
def get_first_x_features(df, day_of_week, include_time, x):
    sub_df = df.iloc[:x, 1:6] #taking columns one to five
    flattened = sub_df.values.flatten()
    if include_time:
        start_time = 1730400000
        minutes_in_day = 60 * 24
        if df.shape[0] < x:
            print("Problem: There are less than 30 features in data")
        startTime = (df.iloc[0, 0] - start_time)/60
        timeOfDay = startTime % minutes_in_day
        timeInTwoPiRange = (2 * math.pi * timeOfDay)/ 1440
        sin_time = math.sin(timeInTwoPiRange)
        cos_time = math.cos(timeInTwoPiRange)
        flattened = np.append(flattened, [sin_time, cos_time])
    if day_of_week != -1:
        flattened = np.append(flattened, day_of_week)

    return flattened



def get_folder_names(prefix = "data/", suffix = "_ohlcv_padded_low_volume_dropped"):
    all_batches = {}

    sep_days_1 = [f"{i:02d}" for i in range(1, 16)]
    sep_days_2 = [f"{i:02d}" for i in range(16, 31)]
    sep_data_1 = [prefix + 'sep' + day + suffix for day in sep_days_1]
    sep_data_2 = [prefix + 'sep' + day + suffix for day in sep_days_2]
    all_batches['sep1'] = sep_data_1
    all_batches['sep2'] = sep_data_2


    oct_days_1 = [f"{i:02d}" for i in range(1, 16)]
    oct_days_2 = [f"{i:02d}" for i in range(16, 32)]
    oct_data_1 = [prefix + 'oct' + day + suffix for day in oct_days_1]
    oct_data_2 = [prefix + 'oct' + day + suffix for day in oct_days_2]
    all_batches['oct1'] = oct_data_1
    all_batches['oct2'] = oct_data_2


    nov_days_1 = [f"{i:02d}" for i in range(1, 16)]
    nov_days_2 = [f"{i:02d}" for i in range(16, 31)]
    nov_data_1 = [prefix + 'nov' + day + suffix for day in nov_days_1]
    nov_data_2 = [prefix + 'nov' + day + suffix for day in nov_days_2]
    all_batches['nov1'] = nov_data_1
    all_batches['nov2'] = nov_data_2


    dec_days_1 = [f"{i:02d}" for i in range(1, 16)]
    dec_days_2 = [f"{i:02d}" for i in range(16, 32)]
    dec_data_1 = [prefix + 'dec' + day + suffix for day in dec_days_1]
    dec_data_2 = [prefix + 'dec' + day + suffix for day in dec_days_2]
    all_batches['dec1'] = dec_data_1
    all_batches['dec2'] = dec_data_2


    jan_days_1 = [f"{i:02d}" for i in range(1, 16)]
    jan_days_2 = [f"{i:02d}" for i in range(16, 32)]
    jan_data_1 = [prefix + 'jan' + day + suffix for day in jan_days_1]
    jan_data_2 = [prefix + 'jan' + day + suffix for day in jan_days_2]
    all_batches['jan1'] = jan_data_1
    all_batches['jan2'] = jan_data_2


    feb_days_1 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    feb_days_2 = ['14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
    feb_data_1 = [prefix + 'feb' + day + suffix for day in feb_days_1]
    feb_data_2 = [prefix + 'feb' + day + suffix for day in feb_days_2]
    all_batches['feb1'] = feb_data_1
    all_batches['feb2'] = feb_data_2

    return all_batches



def continuous_eval(final_open, open_x, change):
    return ((final_open - open_x)/ open_x) * 100


def logistic_eval(final_open, open_x, change):
    label = 0 if final_open <= change * open_x else 1
    return label

def load_only_x_points(csv_files, x):
    X = []
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

        features = get_first_x_features(df, -1, False, x = x)

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


def load_dataset(csv_files, change, x, days_of_week, evalution_func, store_full_df = False, include_day = False, include_time = False):
    X, y, pct_changes = [], [], []
    num_processed = 0
    full_dfs = []
    for day, csv_file in zip(days_of_week, csv_files):
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
        
        if not include_day:
            day = -1
        features = get_first_x_features(df, day, include_time, x = x)
        
        if features is None or df.shape[0] <= x:
            print("Problem: Not Enough Rows in Util")
            print("Problem 1")
            continue

        open_x = df.iloc[x, 1]
        if open_x <= 0:
            continue

        #switched to this rather than final close for clarity
        final_open = df.iloc[-1, 1]

        label = evalution_func(final_open, open_x, change)

        pct_change = ((final_open - open_x)/ open_x) * 100
        #remove
        pct_change = min(pct_change, 500)



        X.append(features)
        y.append(label)
        pct_changes.append(pct_change)

        if store_full_df:
            full_dfs.append(df)

    X = np.array(X)
    y = np.array(y)
    pct_changes = np.array(pct_changes)

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

 

def make_fake_datapoints():
    dp1 = [9.53537473e-05, 1.37633896e-04, 7.66842373e-05,]
    dp2 = [1.00152780e-04, 1.61179280e-04, 1.00152780e-04]
    dp3 = [8.59934543e-05, 8.63070882e-05, 8.56292548e-05]
    return np.array([dp1, dp2, dp3])





def gather_all_csv(folder_names, include_full_path = False):
    all_csv = []
    all_days = []
    for folder in folder_names:
        info = folder.split("/")

        month = info[1][:3]
        day = info[1][3:5]
        day_of_week = get_day_of_the_week(month, day)
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    info = fname.split("_")
                    month = info[0][:3]
                    day = info[0][3:5]
                    all_days.append(day_of_week)
                    all_csv.append(os.path.join(folder, fname))
    return all_csv, all_days

def clear_screen():
    for i in range(300):
        print()



    



