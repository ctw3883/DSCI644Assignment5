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
from model_functions import import_csv, extract_wine_year_from_title, drop_nas, create_test_train_split

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
    return len(create_test_train_split(Xtest, ytest, 0.25)) == 4
    