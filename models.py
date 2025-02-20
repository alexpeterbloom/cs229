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

ALL_MODELS = ONLYCONT + ONLYLOG





#tests models one by one. 
# Change is minimum change needed to mark data for logistic models as 1
#continuous is whether our predictions where continuous
#eval data is function to label y data based on whether model is logistic or continuous

def test_individual_models(models, train_folders, test_folders, change, eval_data, first_min):
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train, y_train, pct_train = load_dataset(train_csvs, change, first_min, eval_data)
    X_test,  y_test,  pct_test  = load_dataset(test_csvs, change, first_min, eval_data)


    print(f"Average % change (Train): {average_percent_change(pct_train):.2f}%")
    print(f"Average % change (Test) : {average_percent_change(pct_test):.2f}%\n")

    for modelInfo in models:
        model = modelInfo[0]
        model_name = modelInfo[1]
        continuous = modelInfo[2]
        print("-" * 50)
        print(f"Model: {model_name}")
        preds = train_model_and_pred(X_train, y_train, X_test, model)
        evaluate(preds, y_test, pct_test, continuous)
        print()
        print()
        print()





def cont_and_logistic_voting(models, train_folders, test_folders, change, first_min):
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train_cont, y_train_cont, pct_train = load_dataset(train_csvs, change, first_min, continuous_eval)

    X_train_log, y_train_log, pct_train = load_dataset(train_csvs, change, first_min, logistic_eval)
    X_test,  y_test,  pct_test = load_dataset(test_csvs, change, first_min, logistic_eval)   
   
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



    preds = combine_logistic_continuous_preds(logistic_predictions, continuous_predictions, "Total_Sum")


    evaluate(preds, y_test, pct_test)


    print()
    print()
    print()






def main():


    allModels = ONLYCONT + ONLYLOG



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

    first_minute = 
    print(f"Running on {first_minute} minutes")


    for split in splits:
        train_folders = split[0]
        test_folders = split[1]
        #print("Train Data")
        #print(train_folders)
        #print("Test Data")
        #print(test_folders)
        print()
        cont_and_logistic_voting(allModels, train_folders, test_folders, 1, first_minute)


main()