# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:31:20 2018

@author: Aldo Contreras
"""

import pandas as pd

dataset = pd.read_csv('myData.csv')
X = dataset.iloc[:, 12:22].values	# flex sensor dataset
y = dataset.iloc[:, 5:9].values		# IMU Quat

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Output matrix conversion
#y_train = y_train.reshape(-1,1)
#y_test = y_test.reshape(-1, 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = sc.fit_transform(y_train) 
y_test = sc.transform(y_test)

# Import the Keras libraries and package
from keras.models import Sequential
from keras.layers import Dense

# building model
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=10, units=64, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=32, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=8, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=4, kernel_initializer="uniform"))
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Fitting the ANN to the training set
results = classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_final = sc.inverse_transform(y_pred)