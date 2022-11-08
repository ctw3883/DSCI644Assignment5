# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:36:28 2022

@author: codyt
"""
import os 
import pandas as pd
import numpy as np 
import sys
path = 'C:\\Users\\codyt\\Documents\\DSCI 644 - Software Engineering for Data Science\\Assignment 5\\'

sys.path.append(path)
from model_functions import import_csv, extract_wine_year_from_title, drop_nas

from sklearn.model_selection import train_test_split
#For using Ensemble model - Adaboost
from sklearn.ensemble import AdaBoostClassifier
# For Finding otimum hyperparameter on Adaboost
from sklearn.model_selection import GridSearchCV
# For Diagnosis of classification model
from sklearn.metrics import classification_report

from sklearn.preprocessing import OrdinalEncoder

data = import_csv(path + 'winemag-data-130k-v2.csv')
data.isnull().sum()

data['wineYear'] = extract_wine_year_from_title(data)
data = drop_nas(data, 'wineYear')
data = drop_nas(data, 'country')
data = drop_nas(data, 'variety')

data['points'] = pd.to_numeric(data['points'])
data['price'] = pd.to_numeric(data['price'])

data = data[['points','country','price','province','taster_name', 'variety', 'winery', 'wineYear']]
data['taster_name'] = data['taster_name'].fillna('')
data['price'] = data['price'].fillna(data['price'].median())

y = data['points']
X = data.drop(['points'], axis = 1)
X0 = pd.get_dummies(X)

X0 = X[['country','province','variety','price','wineYear','winery']]
# X0 = pd.get_dummies(X0)


enc = OrdinalEncoder()
enc.fit(X0)
X1 = enc.fit_transform(X0)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20, random_state=751)

clf=AdaBoostClassifier(random_state=999)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)

clf.feature_importances_


