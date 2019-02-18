#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 05:07:02 2019

@author: Joshua
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Remove The First Dummy Variable To Avoid The Dummy Variable Trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_prediction = regressor.predict(X_test)



import statsmodels.formula.api as sm
#Add Column Of Ones As Statsmodels Does Not Include B0 values
X = np.append(np.ones((50, 1)).astype(int), values = X, axis = 1)
#Use OLS To Determine Which Variable Has The Highest P Value
X_optimum = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimum).fit()
regressor_OLS.summary()
#Remove The Variable Until All Variables Are Below A Significance Limit Of 0.05
X_optimum = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimum).fit()
regressor_OLS.summary()
X_optimum = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimum).fit()
regressor_OLS.summary()
X_optimum = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimum).fit()
regressor_OLS.summary()
X_optimum = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimum).fit()
regressor_OLS.summary()
#Determined That The Administration Spend Was The Significant Variable


