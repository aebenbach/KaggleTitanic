#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:57:19 2022

@author: Andrew Ebenbach

This program tests the model and creates the results CSV for Kaggle's
ML titanic competition
"""

#import test data

import pandas as pd

path_to_file = "/Users/Andrew1/Desktop/Kaggle/Titanic/test.csv"

test_data = pd.read_csv(path_to_file)

from TrainModel import age_mean

#%%

#create new variable categories
test_data["Age"].fillna(age_mean,inplace=True)

test_data["SexBin"] = 1

test_data.loc[test_data.Sex == 'male',"SexBin"] = 0

test_data["BigFam"] = 0

test_data.loc[(test_data.Parch + test_data.SibSp) >= 4, "BigFam"] = 1

#test_data["HasCabin"] = 1

#test_data.loc[test_data.Cabin.isnull(), "HasCabin"] = 0

test_data["IsChild"] = 0

test_data.loc[test_data.Age <= 15 , "IsChild"] = 1


#%%
#create x test dataframe
xvar = pd.DataFrame()

xvar["Pclass"] = test_data.Pclass
#xvar["Age"] = test_data.Age
xvar["Sex"] = test_data.SexBin
xvar["BigFam"] = test_data.BigFam
#xvar["HasCabin"] = test_data.HasCabin
xvar["IsChild"] = test_data.IsChild


#%%
#create predictions
from TrainModel import new_model

ys = pd.DataFrame()

ys = new_model.predict(xvar)

#%%

#create results CSV
results = pd.DataFrame()
results["PassengerId"] = test_data["PassengerId"]
results["Survived"] = ys

results.to_csv("/Users/Andrew1/Desktop/Kaggle/Titanic/SimpleLogRegression.csv", index=False)