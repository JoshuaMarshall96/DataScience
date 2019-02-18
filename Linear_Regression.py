#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 04:06:33 2019

@author: Joshua
"""

import pandas as pd
import matplotlib.pyplot as plt

#Import The Dataset
dataframe = pd.read_csv("Salary_Data.csv")
X = dataframe.iloc[:, :-1].values
Y = dataframe.iloc[:, 1].values

#Create a 30:70 Split As There Are 30 Samples For a Clean Divide
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3)


#Use Simple Linear Regression As A Direct Mapping Between Variable & Outcome Can Be Made
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Create A Prediction From Training Set To Compare With Test Set
y_prediction = regressor.predict(X_test)


#Function To Display Regression Model
def create_Graph(x, y):
    plt.scatter(x, y, color = "red")
    plt.plot(X_train, regressor.predict(X_train), color = "blue")
    plt.title("Salary vs Experience (Training Set)")
    plt.xlabel("Years Experience")
    plt.ylabel("Salary")
    plt.show()


#Applied To Training Set
create_Graph(X_train, Y_train)

#Applied To Test Set
create_Graph(X_test, Y_test)



