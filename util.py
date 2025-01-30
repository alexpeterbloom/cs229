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



#util: contains utility functions used in many files



#optimize: change which features you're taking
#optimize: change how many minutes in the beginning you're taking
def get_first_x_features(df, x = 30):
    if df.shape[0] < x:
        print("Problem: There are less than 30 features in data")
    sub_df = df.iloc[:x, 1:6] #taking columns one to five
    return sub_df.values.flatten()

def load_dataset(csv_files, change, x = 30):
    X, y, pct_changes = [], []
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

        features = get_first_x_features(df, x= x)
        if features is None or df.shape[0] <= x:
            print("Problem: Not Enough Rows in Util")
            continue

        open_x = df.iloc[x, 1]
        if open_x <= 0:
            continue

        #switched to this rather than final close for clarity
        final_open = df.iloc[-1, 1]

        label = 0 if final_open <= change * open_x else 1

        pct_change = ((final_close - open_x)/ open_x) * 100

        X.append(features)
        y.append(label)
        pct_changes.append(pct_change)

    X = np.array(X)
    y = np.array(y)
    pct_changes = np.array(pct_changes)

    print(f"Total CSV files processed: {num_processed}")
    print(f"Total in dataset: {len(X)}")

    return X, y, pct_changes

def average_percent_change(pct_changes):
    return float(np.mean(pct_changes))


def evaluate(preds, y_test):
    cm = confusion_matrix(y_test, preds)
    if cm.shape == (1, 1):
        print("Test set was all predicted to be same thing")
        print(cm)
        return
    
    tn, fp, fn, tp = cm.ravel()
    neg_pred_acc = (tn)/(tn + fn)
    pos_pred_acc = (tp)/(tp + fp)
    print(f'{round(100 * neg_pred_acc)}% of our 0 preds were correct')
    print(f'{round(100 * pos_pred_acc)}% of our 1 preds were correct')

    total_test = len(y_test)
    num_zeros = np.sum(y_test == 0)


#both 
def train_and_evaluate(X_train, y_train, X_test, y_test, pct_test, model):
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

 


    



