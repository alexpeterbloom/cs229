from util import *
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")

warnings.filterwarnings("ignore", category=RuntimeWarning)




ONLYCONT = [
    [RandomForestRegressor(n_estimators=100), 'Random Forest', 1],
    [HistGradientBoostingRegressor(max_iter=100), "HistGradientBoost", 1],
    [XGBRegressor(n_estimators=100, verbosity=0), "XGB", 1],           
    [LGBMRegressor(n_estimators=100, verbose=-1), 'LGBM', 1]       
]

ONLYLOG = [
    [AdaBoostClassifier(n_estimators=50), "AdaBoost", 0],
    [ExtraTreesClassifier(n_estimators=100), "ExtraTrees", 0],
    [HistGradientBoostingClassifier(max_iter=100), "HistGradient", 0],
    [BernoulliNB(), "Bernoulli", 0],
    [XGBClassifier(n_estimators=100, verbosity=0), 'XGB', 0],      
    [LGBMClassifier(n_estimators=100, verbose=-1), 'LGBM', 0],   
    [CatBoostClassifier(iterations=100, logging_level='Silent'), 'CatBoost', 0]  
]

ALL_MODELS = ONLYCONT + ONLYLOG




def one_run_cont_and_log(models, train_folders, test_folders, change, first_min, threshold, logistic_weight):
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train_cont, y_train_cont, pct_train = load_dataset(train_csvs, change, first_min, continuous_eval, silent = True)


    X_train_log, y_train_log, pct_train = load_dataset(train_csvs, change, first_min, logistic_eval, silent = True)
    X_test,  y_test,  pct_test = load_dataset(test_csvs, change, first_min, logistic_eval, silent = True)   


    continuous_predictions = [[] for i in range(len(X_test))]
    logistic_predictions = [[] for i in range(len(X_test))]
    
    for modelInfo in models:
        model = modelInfo[0]
        continuous = modelInfo[2]
        if continuous:
            individual_preds = train_model_and_pred(X_train_cont, y_train_cont, X_test, model)
            for index, p in enumerate(individual_preds):
                continuous_predictions[index].append(p)
        else:
            individual_preds = train_model_and_pred(X_train_log, y_train_log, X_test, model)
            for index, p in enumerate(individual_preds):
                logistic_predictions[index].append(p)


    preds = combine_logistic_continuous_preds(logistic_predictions, continuous_predictions, "Total_Sum", threshold, logistic_weight)


    average_change, count = quiet_eval(preds, y_test, pct_test)
    return average_change, count



def grid_search(models, train_test_splits, first_min, change, thresholds, log_weights):
    for threshold in thresholds:
        for log_weight in log_weights:
            tot_change = 0
            tot_count = 0
            for split in train_test_splits:
                avg_change, count = one_run_cont_and_log(models, split[0], split[1], change, first_min, threshold, log_weight)
                tot_change += count * avg_change
                tot_count += count
                break #change
            error = tot_change/tot_count
            print(f'Threshold: {threshold}, Log_Weight: {log_weight}, Error: {error}, Count: {tot_count}')







def main():


    clear_screen()
    

    january_train1 = ["data/jan0" + str(i) + "_ohlcv_padded_low_volume_dropped" for i in range(1, 10)]
    january_train2 = ["data/jan" + str(i) + "_ohlcv_padded_low_volume_dropped" for i in range(10, 16)]
    january_train = january_train1 + january_train2


    january_test = ["data/jan" + str(i) + "_ohlcv_padded_low_volume_dropped" for i in range(16, 30)]
    january_test.remove("data/jan18_ohlcv_padded_low_volume_dropped")


    dec1 = ["data/dec0" + str(i) + "_ohlcv_padded_low_volume_dropped" for i in range(1, 10)]
    dec2 = ["data/dec" + str(i) + "_ohlcv_padded_low_volume_dropped" for i in range(10, 32)]
    dec = dec1 + dec2
    nov = ["data/nov0" + str(i) + "_ohlcv_padded_low_volume_dropped" for i in range(1, 10)]
    nov.append("data/nov10_ohlcv_padded_low_volume_dropped")

    train_folders = nov + dec + january_test
    test_folders = january_train

    for train in train_folders:
        if train in test_folders:
            print("PANIC TRAINING AND TESTING OVERLAPPING")




    print('beginning baseline')
    
    splits = [[january_train, january_test], [dec2  + january_train, january_test], [dec1 + dec2  + january_train, january_test], [nov + dec1 + dec2 + january_train, january_test]]

    first_minute = 30
    print(f"Running on {first_minute} minutes")




    thresholds = [-20, -10, 0, 10, 20, 30, 40, 50, 60]
    log_weights = [0, 10, 25, 50, 100, 200, 500]


    grid_search(ALL_MODELS, splits, first_minute, 1, thresholds, log_weights)

main()