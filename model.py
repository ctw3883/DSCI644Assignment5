# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:36:28 2022

@author: codyt
"""
import os 
import pandas as pd
import numpy as np 
import sys
paths = ['C:\\Users\\codyt\\Documents\\DSCI 644 - Software Engineering for Data Science\\Assignment 5\\',
         'C:\\Users\\cwilson\\Downloads\\DSCI644Assignment5-main\\DSCI644Assignment5-main\\']
[sys.path.append(path) for path in paths]

from model_functions import import_csv, extract_wine_year_from_title, drop_nas, create_test_train_split, get_target, select_variables, train_model, grid_optimize_model

from sklearn.model_selection import train_test_split
#For using Ensemble model - Adaboost
from sklearn.ensemble import AdaBoostClassifier
# For Finding otimum hyperparameter on Adaboost
from sklearn.model_selection import GridSearchCV
# For Diagnosis of classification model
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from sklearn import tree

from sklearn.preprocessing import OrdinalEncoder

# for my work computer or personal computer 
try:
    data = import_csv(paths[0] + 'winemag-data-130k-v2.csv')
except:
    data = import_csv(paths[1] + 'winemag-data-130k-v2.csv')


data.isnull().sum()
# extract the year of the wine out of the title
data['wineYear'] = extract_wine_year_from_title(data)
# drop any records without wine year
data = drop_nas(data, 'wineYear')
# drop any records without a country
data = drop_nas(data, 'country')
# drop any records without a variety
data = drop_nas(data, 'variety')
# conert the points to a number
data['points'] = pd.to_numeric(data['points'])
# convert the price to a number
data['price'] = pd.to_numeric(data['price'])
# replace any empty tasters with blank string
data['taster_name'] = data['taster_name'].fillna('')
#replace any empty region_1 with blank string
data['region_1'] = data['region_1'].fillna('')
# replace any price with median price
data['price'] = data['price'].fillna(data['price'].median())

# list the vars we want to keep 
my_vars = ['points','country','price','province','region_1','taster_name', 'variety', 'winery', 'wineYear']
my_vars2 = ['points','province','variety','price','wineYear']
# keep only those variables
df = select_variables(data, my_vars)

# get the target & predictors
y, X0 = get_target(df, 'points')

enc = OrdinalEncoder()
enc.fit(X0)
X1 = enc.fit_transform(X0)
# enc.inverse_transform(X1)

test_train = create_test_train_split(X1, y, 0.25)

X_train = test_train[0]
X_test = test_train[1]
y_train = test_train[2]
y_test = test_train[3]



clf = train_model(X_train, y_train)

print('Base model score: ', clf.score(X_test, y_test))

# clf.get_params().keys()

clf_optimized = grid_optimize_model(clf, X_train, y_train)


clf_optimized.score(X_test, y_test)

clf_optimized.feature_importances_

features = pd.Series(clf_optimized.feature_importances_, index=X0.columns)
features.sort_values(ascending=False)

y_train_predicted = clf_optimized.predict(X_train)

y_test_predicted = clf_optimized.predict(X_test)

print(r2_score(y_train, y_train_predicted))

print(mean_absolute_percentage_error(y_train, y_train_predicted))


