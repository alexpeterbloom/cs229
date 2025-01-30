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
    


def main():
    models = {
        "Logistic Regression": LogisticRegression(), 
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),  
        "Support Vector Classifier": SVC(),
        "Naive Bayes": GaussianNB()}
    
    trainFolders = ["jan" + i + "_ohlcv" for i in range(1, 10)]
    testFolders = ['jan' + i + "_ohlcv" for i in range(10, 16)]

    test_individual_models(models, train_folders, test_folders, 1)

main