#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:04:46 2022

@author: Andrew Ebenbach

This program imports and builds a predictive model for Kaggle's ML Titanic
competition. This logisitc regression model predicts the survival of passenger's of the titanic
from 5 variables: Gender, Ticket Class, Childhood,and Family Size. 

The rationale is the following: women and children were prioritized by the crew filling
lifeboats. The higher class tickets were on the upper decks and were able to make
it to the lifeboats in time. Families that were too large became encumbered
and could not make it. 
"""

#import data

import pandas as pd

path_to_file = "/Users/Andrew1/Desktop/Kaggle/Titanic/train.csv"

data = pd.read_csv(path_to_file)

#%%
#create new categories

#female is 1 male is 0
data["SexBin"] = 1

data.loc[data.Sex == 'male',"SexBin"] = 0

#family size >= 4 is 1
data["BigFam"] = 0

data.loc[(data.Parch + data.SibSp) >= 4, "BigFam"] = 1

#replace missing age vals with mean age
age_mean = data["Age"].mean()
data["Age"].fillna(age_mean,inplace=True)

#if cabin is reported the hascabin=1
#data["HasCabin"] = 1

#data.loc[data.Cabin.isnull(), "HasCabin"] = 0

#if age is less than 15 than IsChild=1
data["IsChild"] = 0

data.loc[data.Age <= 15 , "IsChild"] = 1


#%%

#create data frame with x training variables
xvar = pd.DataFrame()

xvar["Pclass"] = data.Pclass
#xvar["Age"] = data.Age
xvar["Sex"] = data.SexBin
xvar["BigFam"] = data.BigFam
#xvar["HasCabin"] = data.HasCabin
xvar["IsChild"] = data.IsChild

#%%

#train model
from sklearn.linear_model import LogisticRegression

new_model = LogisticRegression()

new_model.fit(xvar,data.Survived)

