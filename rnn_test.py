# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:00:12 2018

@author: Aldo Contreras
https://www.superdatascience.com/deep-learning/
"""

# Importing the libraries
import usrInput

def classifierRnn(X, y):
    # Prepare data
    import numpy as np
    training_set = np.append(y, X, axis = 1)

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0.2, 0.8))
    training_set_scaled = sc.fit_transform(training_set)
    flexSens_set_scaled = sc.fit_transform(X)
    IMUquat_set_scaled  = sc.fit_transform(y)

    # Creating a data structure with X timesteps and 1 output (can be other number, based on experience)
    X_train = []
    y_train = []
    timeSteps = 2
    numSamples = len(training_set_scaled)
    numData = len(training_set_scaled[0])
    nInputs = len(X[0])
    nOutputs = len(y[0])
    for nums in range(numData):
        for i in range(timeSteps, numSamples):
            X_train.append(training_set_scaled[(i-timeSteps):i, nums])
            y_train.append(training_set_scaled[i, nums])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    X_test = X_train
    predicted = regressor.predict(X_test)


# Part 3 - Making the predictions and visualising the results
def predictions():
    import pandas as pd
    dataset_train = pd.read_csv('myData.csv')
    real_sensor_values = dataset_train.iloc[:, 12:22].values      # flex sensor dataset

def main():
    [input, target] = usrInput.getDataset()
    classifierRnn(X, y)


if __name__ == "__main__":
    main()
