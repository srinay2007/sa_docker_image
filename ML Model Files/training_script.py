# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:53:00 2023

"""
import pandas as pd
import joblib
import os

os.chdir(
    "C://Users/sri/Documents/2023/Data Science Training/My Data Science Course Slides/23_ML Ops 1/ML project example/"
)
from data_processing_and_features import (
    text_data_cleaning,
    tfidf_features_fit,
    tfidf_features_transform,
)
from model_building import (
    train_test_split,
    fit_and_evaluate_model,
    get_important_features,
)

data = pd.read_csv("data_train.csv")

data_orig = data.copy()

data = text_data_cleaning(data)

data_train = data[0:40000]
data_test = data[40000:]

tfidf, data_train_matrix = tfidf_features_fit(data_train)
features = tfidf.get_feature_names_out()

data_test_matrix = tfidf_features_transform(tfidf, data_test)

x_train, x_test, y_train, y_test = train_test_split(
    data_train, data_test, data_train_matrix, data_test_matrix
)


model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)

feature_importance = get_important_features(model, features)


joblib.dump(model, "model_classifier.pkl")
joblib.dump(tfidf, "tfidf.pkl")


# Improvement scope
# lemmatization, stemming
# word2vec features instead of tfidf
# SHAP feature explanations
# feature selection
