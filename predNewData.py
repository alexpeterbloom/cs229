from util import *
import os

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#
TOTAL_VOTES_NEEDED = 4


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
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train_cont, y_train_cont, pct_train = load_dataset(train_csvs, change, first_min, continuous_eval)
    X_train_log,  y_train_log,  pct_train  = load_dataset(test_csvs, change, first_min, logistic_eval)

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

    preds = combine_logistic_continuous_preds(logistic_predictions, continuous_predictions, "Both_Confirm")

    predicted_csvs = [csv for csv, pred in zip(test_csvs, preds) if pred == 1]
    print("Printing all CSVS we predicted")
    for csv in predicted_csvs:
        print(csv)

   






def main():

    clear_screen()
    ALL_MODELS = ONLYCONT + ONLYLOG

    
    data = get_folder_names()

    train_folders = data['dec1'] + data['dec2'] + data['jan1']
    test_folders = data['jan2']
    confirm_no_overlap(train_folders, test_folders)

    
    minutes = 30
    change = 1


    pred_new_data(ALL_MODELS, train_folders, test_folders, change, minutes)



    
main()