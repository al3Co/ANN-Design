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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

# Import the Keras libraries and package
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# building model
classifier = Sequential()

#v1
classifier.add(Dense(activation="relu", input_dim=10, units=64, kernel_initializer="uniform"))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(activation="relu", units=32, kernel_initializer="uniform"))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(activation="relu", units=8, kernel_initializer="uniform"))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(activation="sigmoid", units=4, kernel_initializer="uniform"))
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

#v2
#classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
#classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae', 'acc'])


classifier.fit(X_train, y_train, batch_size = 10, epochs = 500)

# testing
y_pred = classifier.predict(X_test)
y_pred_inv = sc.inverse_transform(y_pred)

# Visualising the results
import matplotlib.pyplot as plt

plt.plot(y_test, color = 'red', label = 'y real')
plt.plot(y_pred, color = 'blue', label = 'y Predicted')
plt.title('Prediction')
plt.xlabel('Sample')
plt.ylabel('Quaternion')
plt.legend()
plt.show()
