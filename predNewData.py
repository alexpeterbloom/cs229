from util import *
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#
TOTAL_VOTES_NEEDED = 2


ONLYCONT = [
        [RandomForestRegressor(n_estimators=100), 'Random Forest', 1],
        [HistGradientBoostingRegressor(max_iter=100), "HistGradientBoost", 1],
        [XGBRegressor(n_estimators=100),"XGB", 1],
        [LGBMRegressor(n_estimators=100), 'LGBM', 1]
]
    
ONLYLOG = [
        [AdaBoostClassifier(n_estimators=50), "AdaBoost", 0],
        [ExtraTreesClassifier(n_estimators=100), "ExtraTrees", 0],
        [HistGradientBoostingClassifier(max_iter=100), "HistGradient", 0],
        [BernoulliNB(), "Bernoulli", 0],
        [XGBClassifier(n_estimators=100), 'XGB', 0],
        [LGBMClassifier(n_estimators=100),'LGBM', 0],
        [CatBoostClassifier(iterations=100, verbose=False) , 'CatBoost', 0]
]


def mixed_pred_func(all_preds):
    total_sum = 0
    average_abs = 0
    count = 0
    for pred in all_preds:
        if pred == 1:
            total_sum += 50
        elif pred == 0:
            total_sum -= 50
        else:
            total_sum += pred
            average_abs += abs(pred)
            count += 1
    if (total_sum/len(all_preds)) > 35:
        return 1
    return 0



def logisticVoting(all_preds):
    ones = all_preds.count(1)
    if ones >= TOTAL_VOTES_NEEDED:
        return 1
    return 0


def continuousVoting(all_preds):
    pred = sum(all_preds)/len(all_preds)
    return 1 if pred > 30 else 0



def pred_new_data(models, train_folders, test_folders, change, first_min):
    train_csvs, days_of_week_train = gather_all_csv(train_folders)
    test_csvs, days_of_week_test = gather_all_csv(test_folders)


    X_train_cont, y_train_cont, pct_train, datasets = load_dataset(train_csvs, change, first_min, days_of_week_train, continuous_eval, True, False, False)
    
    X_train_log, y_train_log, pct_train, datasets = load_dataset(train_csvs, change, first_min, days_of_week_train, logistic_eval, True, False, False)
    X_test = load_only_x_points(test_csvs, first_min)
   
    print(f"Average % change (Train): {average_percent_change(pct_train):.2f}%")

    continuous_predictions = [[] for i in range(len(X_test))]
    logistic_predictions = [[] for i in range(len(X_test))]
    for modelInfo in models:
        model = modelInfo[0]
        name = modelInfo[1]
        continuous = modelInfo[2]
        print("-" * 50)
        print(f"Model: {name}")
        if continuous:
            individual_preds = train_model_and_pred(X_train_cont, y_train_cont, X_test, model)
            for index, p in enumerate(individual_preds):
                continuous_predictions[index].append(p)
        else:
            individual_preds = train_model_and_pred(X_train_log, y_train_log, X_test, model)
            for index, p in enumerate(individual_preds):
                logistic_predictions[index].append(p)


    if len(logistic_predictions[0]) > 0:
        logistic_predicted = [1 if logisticVoting(all_preds) == 1 else 0 for all_preds in logistic_predictions]
        print(f'Logistic predicted {logistic_predicted.count(1)}')
    else:
        print("Error: No Logistic Models Present")

    if len(continuous_predictions[0]) > 0: #we did have some continuous models
        continuous_predicted = [1 if continuousVoting(all_preds) == 1 else 0 for all_preds in continuous_predictions]
        print(f'Continuous predicted {continuous_predicted.count(1)}')
    else:
        print("Error: No Continuous Models Present")




    preds= []
    for l, c in zip(logistic_predicted, continuous_predicted):
        if l and c:
            preds.append(1)
        else:
            preds.append(0)

    predicted_csvs = [csv for csv, pred in zip(test_csvs, preds) if pred == 1]
    print("Printing all CSVS we predicted")
    for csv in predicted_csvs:
        print(csv)

   






def main():

    clear_screen()
    ALL_MODELS = ONLYCONT + ONLYLOG

    
    january_train1 = ["data/jan0" + str(i) + "_ohlcv_padded" for i in range(1, 10)]
    january_train2 = ["data/jan" + str(i) + "_ohlcv_padded" for i in range(10, 16)]
    january_train = january_train1 + january_train2


    january_test = ["data/jan" + str(i) + "_ohlcv_padded" for i in range(16, 30)]
    january_test.remove("data/jan18_ohlcv_padded")


    dec1 = ["data/dec0" + str(i) + "_ohlcv_padded" for i in range(1, 10)]
    dec2 = ["data/dec" + str(i) + "_ohlcv_padded" for i in range(10, 32)]
    dec = dec1 + dec2
    nov = ["data/nov0" + str(i) + "_ohlcv_padded" for i in range(1, 10)]
    nov.append("data/nov10_ohlcv_padded")



    start_addr = "data/_do_not_train_on/"
    end_addr = "_ohlcv_padded_first30"
    dates = ['jan30', 'jan31', 'feb01', 'feb02', 'feb03', 'feb04', 'feb05', 'feb06', 'feb07'
             'feb08', 'feb09', 'feb10', 'feb11', 'feb12', 'feb13']
    
    final_test = [start_addr + date + end_addr for date in dates]

    train_folders = dec + january_test + january_train
    test_folders = final_test

    print('final train folders', train_folders)

    print()
    print('final test folders', test_folders)



    pred_new_data(ALL_MODELS, train_folders, test_folders, 1, 2)



    
main()