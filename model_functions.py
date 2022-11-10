# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:36:28 2022

@author: codyt
"""
import os 
import pandas as pd
import numpy as np 
import sys
sys.path.append('C:\\Users\\codyt\\Documents\\DSCI 644 - Software Engineering for Data Science\\Assignment 5\\')
sys.path.append('C:\\Users\\cwilson\\Downloads\\DSCI644Assignment5-main\\DSCI644Assignment5-main\\')


from sklearn.model_selection import train_test_split
#For using Ensemble model - Adaboost
from sklearn.ensemble import AdaBoostClassifier
# For Finding otimum hyperparameter on Adaboost
from sklearn.model_selection import GridSearchCV
# For Diagnosis of classification model
from sklearn.metrics import classification_report
from sklearn import tree

def import_csv(filepath):
    return pd.read_csv(filepath, index_col=0)

def extract_wine_year_from_title(df):
    s = df['title'].str.extract('([1|2][0-9][0-9][0-9])')
    return pd.to_numeric(s[0])

def drop_nas(df, col):
    return df[~df[col].isna()]

def create_test_train_split(X_df, y_series, split_size):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=split_size, random_state=751)
    return [X_train, X_test, y_train, y_test]

def get_target(df, target_col):
    return df[target_col], df.drop([target_col], axis=1)

def select_variables(df, vars_list):
    return df[vars_list]

def train_model(X_train, y_train):
        
    clf = tree.DecisionTreeRegressor()
    
    clf.fit(X_train, y_train)
    
    return clf
