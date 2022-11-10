# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:13:10 2022

@author: codyt
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('C:\\Users\\codyt\\Documents\\DSCI 644 - Software Engineering for Data Science\\Assignment 5\\')
sys.path.append('C:\\Users\\cwilson\\Downloads\\DSCI644Assignment5-main\\DSCI644Assignment5-main\\')
from model_functions import import_csv, extract_wine_year_from_title, drop_nas, create_test_train_split, get_target, select_variables, train_model

from sklearn import tree

def test_import_csv_fail():
    filepath = 'C:\\Users\\codyt\\Documents\\DSCI 644 - Software Engineering for Data Science\\Assignment 5\\winemag-data-130k-v2.csv'
    assert isinstance(import_csv(filepath), pd.Series)
    

def test_import_csv_succeed():
    filepath = 'C:\\Users\\codyt\\Documents\\DSCI 644 - Software Engineering for Data Science\\Assignment 5\\winemag-data-130k-v2.csv'
    assert isinstance(import_csv(filepath), pd.DataFrame)
    
    
def test_extract_wine_year_fail():
    # the function only looks for 4 digits in a row starting with 1 or 2, not 3
    tester = pd.DataFrame(columns=['title'], data=['2014', 'a cow in 1998', 'once in 1777', 'the year 3004'])
    assert extract_wine_year_from_title(tester).isnull().sum() == 0

def test_extract_wine_year_succeed():
    tester = pd.DataFrame(columns=['title'], data=['2014', 'a cow in 1998', 'once in 1777', 'the year 3004'])
    assert extract_wine_year_from_title(tester).isnull().sum() == 1
    
def test_drop_nas_fail():
    tester = pd.DataFrame(columns=['dummy'], data=['cow', np.nan,'dog'])
    assert tester.shape[0] == drop_nas(tester, 'dummy').shape[0]
    
def test_drop_nas_fail1():
    tester = pd.DataFrame(columns=['dummy'], data=['cow', np.nan,'dog', np.nan])
    assert drop_nas(tester, 'dummy').shape[0] == 4
    
def test_drop_nas_succeed():
    tester = pd.DataFrame(columns=['dummy'], data=['cow', np.nan,'dog'])
    assert tester.shape[0] > drop_nas(tester, 'dummy').shape[0]
    
def test_spliting_succeed():
    Xtest = pd.DataFrame(columns=['dummy'], data=['yes','no','yes','no'])
    ytest = pd.Series(data=['cow','dog','frog','log'])
    assert len(create_test_train_split(Xtest, ytest, 0.25)) == 4
    
def test_get_target_fail():
    tester = pd.DataFrame(columns=['x0','x1','y'], data=[[1,2,3],[1,2,5],[2,4,8]])
    y, X = get_target(tester, 'y')
    assert isinstance(y, pd.DataFrame)
    
def test_get_target_fail1():
    tester = pd.DataFrame(columns=['x0','x1','y'], data=[[1,2,3],[1,2,5],[2,4,8]])
    y, X = get_target(tester, 'y')
    assert y.shape == X.size
    
def test_get_target_succeed():
    tester = pd.DataFrame(columns=['x0','x1','y'], data=[[1,2,3],[1,2,5],[2,4,8]])
    y, X = get_target(tester, 'y')
    assert isinstance(y, pd.Series)  
    
def test_select_variables_fail():
    tester = pd.DataFrame(columns=[0,1,2,3], data=[[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    keep = [0,1]
    assert select_variables(tester, keep).shape[1] != len(keep)
    
def test_select_variables_succeed():
    tester = pd.DataFrame(columns=[0,1,2,3], data=[[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    keep = [0,1]
    assert select_variables(tester, keep).shape[1] == len(keep)
    
def test_train_model_fail():
    X_train = np.array([[1,2,3],[4,5,6],[7,8,9]])
    y_train = np.array([[100],[101],[102]])
    clf = train_model(X_train, y_train)
    assert not isinstance(clf, tree.DecisionTreeRegressor)

def test_train_model_succeed():
    X_train = np.array([[1,2,3],[4,5,6],[7,8,9]])
    y_train = np.array([[100],[101],[102]])
    clf = train_model(X_train, y_train)
    assert not isinstance(clf, tree.DecisionTreeRegressor)