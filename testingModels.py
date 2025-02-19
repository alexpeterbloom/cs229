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


TEST_VOTES = 3
VOTES_NEEDED = 3

CATEGORICAL_MODELS  = {
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "HistGBM": HistGradientBoostingClassifier(max_iter=100),
        "Bernoulli NB": BernoulliNB(),
        "XGBoost": XGBClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=False)
    }

REGRESSION_MODELS = {
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "HistGBM": HistGradientBoostingRegressor(max_iter=100),
    "XGBoost": XGBRegressor(n_estimators=100),
    "LightGBM": LGBMRegressor(n_estimators=100),
    }

THEORY = [[LGBMRegressor(n_estimators=100), 'LGBM', 1]]

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

TEST_MODELS = [
    [HistGradientBoostingRegressor(max_iter=100), "HistGradientBoost", 1],
    [XGBRegressor(n_estimators=100),"XGB", 1],
    [AdaBoostClassifier(n_estimators=50), "AdaBoost", 0],
    [ExtraTreesClassifier(n_estimators=100), "ExtraTrees", 0],
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
    if ones >= VOTES_NEEDED:
        return 1
    return 0


def continuousVoting(all_preds):
    pred = sum(all_preds)/len(all_preds)
    return 1 if pred > 30 else 0


def test_individual_models(models, train_folders, test_folders, change, continuous, eval_data, first_min):

    train_csvs, days_of_week = gather_all_csv(train_folders)
    test_csvs, days_of_week = gather_all_csv(test_folders)

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

def graph_hist(data):
    plt.hist(data, bins=5, edgecolor='black')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Histogram Example')

def contAndLogistic(models, train_folders, test_folders, change, mixed_pred_func, graph, first_min):
    train_csvs, days_of_week_train = gather_all_csv(train_folders)
    test_csvs, days_of_week_test = gather_all_csv(test_folders)

    X_train_cont, y_train_cont, pct_train, datasets = load_dataset(train_csvs, change, first_min, days_of_week_train, continuous_eval, True, False, False)
    
    X_train_log, y_train_log, pct_train, datasets = load_dataset(train_csvs, change, first_min, days_of_week_train, logistic_eval, True, False, False)
    X_test,  y_test,  pct_test, datasets = load_dataset(test_csvs, change, first_min, days_of_week_test, logistic_eval, True, False, False)   
   
    print(f"Average % change (Train): {average_percent_change(pct_train):.2f}%")
    print(f"Average % change (Test) : {average_percent_change(pct_test):.2f}%\n")

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

    if len(continuous_predictions[0]) > 0: #we did have some continuous models
        continuous_predicted = [1 if continuousVoting(all_preds) == 1 else 0 for all_preds in continuous_predictions]
        print(f'Continuous predicted {continuous_predicted.count(1)}')




    preds= []
    for l, c in zip(logistic_predicted, continuous_predicted):
        if l and c:
            preds.append(1)
        else:
            preds.append(0)

    pcts_predicted = []
    for index, pred in enumerate(preds):
        if pred:
            pcts_predicted.append(pct_test[index])
    graph_hist(pcts_predicted)

    evaluate(preds, y_test, pct_test)

    if graph:
        graphTimeSeries(datasets, preds)

    print()
    print()
    print()

def train_and_test_on_fake_data(models, train_folders, change, predict_1_func, eval_data, first_min):
    train_csvs = gather_all_csv(train_folders)
    X_test = make_fake_datapoints()

    X_train, y_train, pct_train, datasets = load_dataset(train_csvs, change, first_min, eval_data, True)


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

    for i in range(len(X_test)):
        print("Our Data Point")
        print((X_test[i]))
        print("Our Pred")
        print((preds[i]))







def averageModels(models, train_folders, test_folders, change, predict_1_func, eval_data, graph, first_min):
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


    evaluate(preds, y_test, pct_test)

    if graph:
        graphTimeSeries(datasets, preds)

    print()
    print()
    print()







def main():


    allModels = ONLYCONT + ONLYLOG



    clear_screen()

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

    train_folders = nov + dec + january_test
    test_folders = january_train

    for train in train_folders:
        if train in test_folders:
            print("PANIC TRAINING AND TESTING OVERLAPPING")


    #print(train_folders)
    #print('')
    #print(test_folders)

    print('beginning baseline')
    
    splits = [[january_train, january_test], [dec2  + january_train, january_test], [dec1 + dec2  + january_train, january_test], [nov + dec1 + dec2 + january_train, january_test]]

    for split in splits:
        train_folders = split[0]
        test_folders = split[1]
        print("Train Data")
        print(train_folders)
        print("Test Data")
        print(test_folders)
        print()
        contAndLogistic(allModels, train_folders, test_folders, 1, mixed_pred_func, False, 2)
        
    #train_and_test_on_fake_data(allModels, dec, 1, mixed_pred_func, eval_data, first_min)
    #contAndLogistic(allModels, dec, january_train, 1, mixed_pred_func, False, 1)
    

main()