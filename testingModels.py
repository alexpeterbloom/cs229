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


VOTES_NEEDED = 4



def logisticVoting(all_preds):
    ones = all_preds.count(1)
    if ones >= VOTES_NEEDED:
        return 1
    return 0


def continuousVoting(all_preds):
    pred = sum(all_preds)/len(all_preds)
    return 1 if pred > 80 else 0


def test_individual_models(models, train_folders, test_folders, change, continuous, eval_data, first_min = 30):

    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train, y_train, pct_train = load_dataset(train_csvs, change, first_min, eval_data)
    X_test,  y_test,  pct_test  = load_dataset(test_csvs, change, first_min, eval_data)



    print(f"Average % change (Train): {average_percent_change(pct_train):.2f}%")
    print(f"Average % change (Test) : {average_percent_change(pct_test):.2f}%\n")

    for model_name, model in models.items():
        print("-" * 50)
        print(f"Model: {model_name}")
        preds = train_model_and_pred(X_train, y_train, X_test, model)
        evaluate(preds, y_test, pct_test, continuous)
        print()
        print()
        print()



    

def averageModels(models, train_folders, test_folders, change, predict_1_func, eval_data, graph, first_min = 30):
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train, y_train, pct_train, datasets = load_dataset(train_csvs, change, first_min, eval_data, True)
    X_test,  y_test,  pct_test, datasets = load_dataset(test_csvs, change, first_min, logistic_eval, True)

    print(f"Average % change (Train): {average_percent_change(pct_train):.2f}%")
    print(f"Average % change (Test) : {average_percent_change(pct_test):.2f}%\n")

    all_predictions = [[] for i in range(len(X_test))]
    for model_name, model in models.items():
        print("-" * 50)
        print(f"Model: {model_name}")
        individual_preds = train_model_and_pred(X_train, y_train, X_test, model)
        for index, p in enumerate(individual_preds):
            all_predictions[index].append(p)

    preds = [1 if predict_1_func(all_preds) == 1 else 0 for all_preds in all_predictions]


    max_up = 0
    total = 0
    total_sum = 0
    for i in range(len(preds)):
        if preds[i] == 1:
            result = pct_test[i]
            total += 1
            total_sum += result
            if result > max_up:
                max_up = result
            print(result)
    if (total == 0):
        print('predicted nothing')
        return
    
    print(f'max: {max_up}, average: {total_sum/total}, count: {total}')
    evaluate(preds, y_test, pct_test)

    if graph:
        graphTimeSeries(datasets, preds)

    print()
    print()
    print()



def main():
    categorical_models = {
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "HistGBM": HistGradientBoostingClassifier(max_iter=100),
        "Bernoulli NB": BernoulliNB(),
        "XGBoost": XGBClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=False)
    }

    regression_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "HistGBM": HistGradientBoostingRegressor(max_iter=100),
    "XGBoost": XGBRegressor(n_estimators=100),
    "LightGBM": LGBMRegressor(n_estimators=100),
    }
    
    clear_screen()

    january = ["data/jan" + str(i) + "_ohlcv_padded" for i in range(1, 16)]


    dec = ["data/dec0" + str(i) + "_ohlcv_padded" for i in range(1, 10)]
    dec2 = ["data/dec" + str(i) + "_ohlcv_padded" for i in range(10, 27)]
    dec = dec + dec2
    nov = ["data/nov0" + str(i) + "_ohlcv_padded" for i in range(1, 10)]
    nov.append("data/nov10_ohlcv_padded")

    train_folders = ["data/nov02_ohlcv_padded"]
    print(train_folder 

    averageModels(categorical_models, train_folders, test_folders, 1, logisticVoting, logistic_eval, graph = True)

main()