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

#util: contains utility functions used in many files


    #month_str = month_str.capitalize()
    
    # Get the current year
    #current_year = datetime.now().year
    
    # Create a date string in the format "Nov 09 2025" (for example)
    #date_str = f"{month_str} {day_str} {current_year}"
    
    # Parse the date string using strptime with the format "%b %d %Y"
    #date_obj = datetime.strptime(date_str, "%b %d %Y")
    
    # datetime.weekday() returns 0 for Monday, 6 for Sunday.
    # Adding 1 adjusts the range to 1 (Monday) through 7 (Sunday).



def get_day_of_the_week(month, day):
    upper_month = month.capitalize()
    months_last_year = ['NOV', 'DEC']
    if upper_month in months_last_year:
        current_year = 2024
    else:
        current_year = 2025
    date_str = f"{upper_month} {day} {current_year}"
    date_obj = datetime.strptime(date_str, "%b %d %Y")
    return date_obj.weekday()



#optimize: change which features you're taking
#optimize: change how many minutes in the beginning you're taking
def get_first_x_features(df, day_of_week, include_time, x = 30):
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




def continuous_eval(final_open, open_x, change):
    return ((final_open - open_x)/ open_x) * 100


def logistic_eval(final_open, open_x, change):
    label = 0 if final_open <= change * open_x else 1
    return label

def load_dataset(csv_files, change, x, day_of_week, evalution_func, store_full_df = False):
    X, y, pct_changes = [], [], []
    num_processed = 0
    full_dfs = []
    for day, csv_file in zip(day_of_week, csv_files):
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

        features = get_first_x_features(df, -1, True, x= x)
        if features is None or df.shape[0] <= x:
            print("Problem: Not Enough Rows in Util")
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

def graphTimeSeries(test_dfs, preds, x = 30):
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

 


def gather_all_csv(folder_names):
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



    



