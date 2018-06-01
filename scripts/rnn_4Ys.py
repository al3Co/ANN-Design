# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:00:12 2018

"""

def createRNN(X, y, batchS, nEpochs, optim_lossT, option):
    # Prepare data
    import numpy as np
    training_set = np.append(y, X, axis = 1)
    # Feature Scaling
    if option == 'reader':
        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0.2, 0.8))
        training_set_scaled = sc.fit_transform(training_set)
    else:
        training_set_scaled = training_set
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
    # properties
    unit = 32
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = unit, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    #regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = unit, return_sequences = True))
    #regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = unit, return_sequences = True))
    #regressor.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = unit))
    #regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    # Compiling the RNN
    regressor.compile(optimizer = optim_lossT[0], loss = optim_lossT[1], metrics=['mae', 'acc'])
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = nEpochs, batch_size = batchS)

    scores = regressor.evaluate(X_train, y_train)
    print('PARAMETERS --> dense units: {0}, optimizer: {1} loss: {2} batch_size: {3} epochs: {4}'
        .format(unit, optim_lossT[0], optim_lossT[1], batchS, nEpochs))
    print("{0}: {1} --> {2}: {3} %"
        .format(regressor.metrics_names[1], scores[1], regressor.metrics_names[2], scores[2]*100))
    X_test = X_train
    y_pred = regressor.predict(X_test)
    return(regressor, scores, unit)

def predictions(): # TODO
    import pandas as pd
    dataset_train = pd.read_csv('myData.csv')
    real_sensor_values = dataset_train.iloc[:, 12:22].values      # flex sensor dataset

def main():
    import usrInput
    [X, y] = usrInput.getDataset()
    batchS, nEpochs = 32, 1
    optim_lossT = ['nadam', 'mean_squared_error']
    option = 'other'
    [regressor, scores, unit] = createRNN(X, y, batchS, nEpochs, optim_lossT, option)

if __name__ == "__main__":
    main()
