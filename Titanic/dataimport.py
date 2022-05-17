#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:04:46 2022

@author: Andrew1
"""


import pandas as pd

path_to_file = "/Users/Andrew1/Desktop/Kaggle/Titanic/train.csv"

data = pd.read_csv(path_to_file)

#%%

data["SexBin"] = 1

data.loc[data.Sex == 'male',"SexBin"] = 0

data["BigFam"] = 0

data.loc[(data.Parch + data.SibSp) >= 4, "BigFam"] = 1

age_mean = data["Age"].mean()
data["Age"].fillna(age_mean,inplace=True)

data["HasCabin"] = 1

data.loc[data.Cabin.isnull(), "HasCabin"] = 0

data["IsChild"] = 0

data.loc[data.Age <= 15 , "IsChild"] = 1


#%%

xvar = pd.DataFrame()

xvar["Pclass"] = data.Pclass
#xvar["Age"] = data.Age
xvar["Sex"] = data.SexBin
xvar["BigFam"] = data.BigFam
xvar["HasCabin"] = data.HasCabin
xvar["IsChild"] = data.IsChild

#%%
from sklearn.linear_model import LogisticRegression

new_model = LogisticRegression()

new_model.fit(xvar,data.Survived)

