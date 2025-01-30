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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def test_individual_models(models, train_folders, test_folders, change):

    train_csvs = gather_all_csv(train_folders)
    test_csvs = gather_all_csv(test_folders)

    X_train, y_train, pct_train = load_dataset(train_csvs, change, x=30)
    X_test,  y_test,  pct_test  = load_dataset(test_csvs, change, x=30)



    print(f"Average % change (Train): {average_percent_change(pct_train):.2f}%")
    print(f"Average % change (Test) : {average_percent_change(pct_test):.2f}%\n")

    for model_name, model in models.items():
        print("-" * 50)
        print(f"Model: {model_name}")
        preds = train_model_and_pred(X_train, y_train, X_test, model)
        evaluate(preds, y_test, pct_test)
        print()
        print()
        print()
    


def main():
    models = {
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "HistGBM": HistGradientBoostingClassifier(max_iter=100),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Bernoulli NB": BernoulliNB(),
        "Multinomial NB": MultinomialNB(),
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": SGDClassifier(loss="hinge"),
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "XGBoost": XGBClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=False)
    }
    
    train_folders = ["data/jan" +str(i) + "_ohlcv_padded" for i in range(1, 10)]
    test_folders = ['data/jan' + str(i) + "_ohlcv_padded" for i in range(10, 16)]

    test_individual_models(models, train_folders, test_folders, 1)

main()