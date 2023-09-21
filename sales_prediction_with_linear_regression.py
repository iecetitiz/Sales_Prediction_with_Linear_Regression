# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:00:29 2023

@author: iecet
"""
#importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

#loading data
df = pd.read_csv("C:/datasets/Advertising.csv")

#taking columns
df.columns

#dropping 'Unnamed: 0' column because we don't need it
df = df.drop('Unnamed: 0', axis = 1)

#taking target variable and other variable
X = df.drop('Sales', axis = 1)
y = df[['Sales']]

#dividing a data set into two parts, train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

#fitting the model
reg_model = LinearRegression().fit(X_train, y_train)

#y_hat = b + w1 * TV + w2 * Radio + w3 * Newspaper

b = reg_model.intercept_[0] #2.907947020816433
w = reg_model.coef_

w1 = reg_model.coef_[0][0] #0.04684310317699042
w2 = reg_model.coef_[0][1] #0.17854434380887607
w3 = reg_model.coef_[0][2] #0.00258618609398899

#making prediction with X_train
y_predict = reg_model.predict(X_train)

#mean squared error
mse = mean_squared_error(y_train, y_predict)

#mean absolute error
mae = mean_absolute_error(y_train, y_predict)

#root mean squared error
rmse = np.sqrt(mse)

#k-fold cross validation
cross_val = cross_val_score(reg_model, X, y, )



























