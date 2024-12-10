# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:44:10 2023

"""
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from xgboost import XGBClassifier


def train_test_split(data_train, 
                     data_test,\
                     data_train_matrix,
                     data_test_matrix):
    
    data_train['sentiment'] = data_train['sentiment'].map({'negative':0, 'positive' : 1})
    data_test['sentiment'] = data_test['sentiment'].map({'negative':0, 'positive' : 1})
    y_train = data_train['sentiment']
    y_test = data_test['sentiment']
    
    x_train = data_train_matrix.copy()
    x_test = data_test_matrix.copy()

    return x_train, x_test, y_train, y_test


def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    xgb =  XGBClassifier(random_state=0)
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
    xgb_acc_score = accuracy_score(y_test, xgb_predict)
    print("confussion matrix")
    print(xgb_conf_matrix)
    print("\n")
    print("Accuracy of XGBoost:",xgb_acc_score*100,'\n')
    print(classification_report(y_test,xgb_predict))
    return xgb

def get_important_features(model, features):
    importances = pd.DataFrame(model.feature_importances_)
    importances['features'] = features
    importances.columns = ['importance','feature']
    importances.sort_values(by = 'importance', ascending= False,inplace=True)

    return importances
