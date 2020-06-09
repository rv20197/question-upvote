#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:54:00 2020

@author: vatsal
"""


# =============================================================================
# %% Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder,Binarizer,StandardScaler
import pickle
# =============================================================================

# =============================================================================
# %% importing Data
train = pd.read_csv("/home/vatsal/Documents/Predict Number of Upvotes/train.csv")
train = train.drop(train[train.Views > 3000000].index)
# =============================================================================

# =============================================================================
# %% Data Preprocessing
LE = LabelEncoder()
train["Tag"] = LE.fit_transform(train["Tag"])
train.drop(["ID","Username"],axis = 1,inplace = True)

bins = Binarizer(threshold = 10)
train["Answers"] = bins.fit_transform(train["Answers"])[0]

x = train[["Tag","Reputation","Answers","Views"]].values
y = train["Upvotes"].values
# =============================================================================

# =============================================================================
# %% Model Building
model = GradientBoostingRegressor()
model.fit(x,y)

pickle.dump(model,open("model.pkl","wb"))
# =============================================================================

test = pd.read_csv("/home/vatsal/Documents/Predict Number of Upvotes/Mantest.csv")
model1 = pickle.load(open('model.pkl', 'rb'))
upvotes = model1.predict(test)
