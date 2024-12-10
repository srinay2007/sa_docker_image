# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:44:32 2023

"""

import os
import joblib
import pandas as pd
os.chdir("C://Users/sri/Documents/2023/Data Science Training/My Data Science Course Slides/23_ML Ops 1/ML project example/")
from data_processing_and_features import text_data_cleaning, tfidf_features_transform

model = joblib.load('model_classifier.pkl')
tfidf = joblib.load('tfidf.pkl')

# in industry you would read this data from some SQL table
data_pred = pd.read_csv("data_pred.csv")
# Use same data cleaning and preprocessing pipeline as testing
data_pred = text_data_cleaning(data_pred)

# Use same feature extraction pipeline as testing
data_pred_matrix = tfidf_features_transform(tfidf, data_pred)

results_label = model.predict(data_pred_matrix)
results_probability = model.predict_proba(data_pred_matrix)
