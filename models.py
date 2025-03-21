from util import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random





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

def test_individual_models(models, train_folders, test_folders, change, eval_data, first_min, feature_names):
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train, train_stat, y_train, pct_train = load_dataset(train_csvs, change, first_min, eval_data, feature_names, [])
    X_test,  test_stat, y_test,  pct_test  = load_dataset(test_csvs, change, first_min, eval_data, feature_names, [])


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





def cont_and_logistic_voting(models, train_folders, test_folders, change, first_min, feature_names):
    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)


    X_train_cont, x_stat, y_train_cont, pct_train = load_dataset(train_csvs, change, first_min, continuous_eval, feature_names, [])


    X_train_log, x_stat, y_train_log, pct_train = load_dataset(train_csvs, change, first_min, logistic_eval, feature_names, [])
    X_test,  x_stat, y_test,  pct_test = load_dataset(test_csvs, change, first_min, logistic_eval, feature_names, [])   
   
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








def main():


    allModels = [[AdaBoostClassifier(n_estimators=50), "AdaBoost", 0]]



    clear_screen()
    

    data = get_folder_names(suffix = "_extra_features")



    train = ['sep', 'oct', 'nov', 'dec', 'jan']
    val = ['feb']

    folder_names_dict = get_folder_names(prefix = "data/", suffix = "_json_padded")
    

    train_folders = []
    for month in train:
        train_folders.append(folder_names_dict[month])

    val_folders = []
    for month in val:
        val_folders.append(folder_names_dict[month])
 

    first_minutes = 30

    confirm_no_overlap(train, val)

    feature_names = ['open','high','low','close','volume']
    test_individual_models(allModels, train_folders, val_folders, CHANGE_NEEDED, logistic_eval, 30, feature_names)

main()