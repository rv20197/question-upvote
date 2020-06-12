#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 07:33:58 2020

@author: vatsal
"""


# =============================================================================
# %% Libraries
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Binarizer
import numpy as np
import pickle as p
# =============================================================================

# =============================================================================
# %% Data Builing
train = pd.read_csv('/home/vatsal/Documents/Hackathons/Predict Number of Upvotes/train.csv')
train.describe()

# Here we'll remove the extreme points from the data
train = train.drop(train[train.Views > 3000000].index)

# Encoding the Categorical Variable
labelencoder_X = LabelEncoder()
train['Tag'] = labelencoder_X.fit_transform(train['Tag'])
p.dump(labelencoder_X, open("LE.pkl","wb"))
# Remove Unnecessary Feature
train.drop(['ID','Username'], axis=1,inplace =True)

# Binner
bn = Binarizer(threshold=10)
pd_watched = bn.transform([train['Answers']])[0]
train['pd_watched'] = pd_watched
p.dump(bn,open("biner.pkl","wb"))

feature_names = [a for a in train.columns if a not in ['Upvotes']]
X = train[feature_names] 
y = train["Upvotes"]

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 205)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
p.dump(sc,open("scaler.pkl","wb"))

poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X_train)
X_test_Poly = poly_feat.transform(X_test)
p.dump(poly_feat,open('Poly.pkl', "wb"))

model = linear_model.LassoLars(alpha = 0.021, max_iter = 150, random_state = 205)
model.fit(X_poly,y_train)
p.dump(model,open('model.pkl','wb'))
preds = model.predict(X_test_Poly)

rmse = np.sqrt(mean_squared_error(y_test,preds))
# =============================================================================

