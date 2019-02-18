#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 02:37:47 2019

@author: Joshua
"""

import pandas as pd

#Import The Dataset
dataframe = pd.read_csv("Data.csv")
X = dataframe.iloc[:, :-1].values
Y = dataframe.iloc[:, -1:].values

#Prepare The Dataset For Useful Data Analysis
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

#Imputer Used To Fill Any Missing Data With The Mean Of The Column 
imputer = Imputer('NaN', 'mean', 0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#LabelEncoder Used To Turn Categorical Data Into Quantitative Data For Machine Learning Algorithms
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:, 0])
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

#OneHotEncoder Creates Dummy Variables In Order To Remove Bias Between Categorical Data Caused By LabelEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


#Create The Training & Test Sets
from sklearn.model_selection import train_test_split

#Train_Test_Split Here produces A Training Set Of 80% Of The Data & A Test Set Of 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


#Feature Scale Variables So That Variables Are All On The Same Scale
from sklearn.preprocessing import StandardScaler

#StandardScaler Only Applied On X As Y Is Categorical
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



