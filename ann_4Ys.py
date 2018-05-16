# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:05:10 2018

@author: Aldo Contreras
"""
# Importing the libraries
import matplotlib.pyplot as plt
import usrInput

# Data Preprocessing
def importAndPrepare(X, y):
    # Importing sklearn tools for prepare data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0.2, 0.8))
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    X_test = sc.fit_transform(X_test)
    y_test = sc.fit_transform(y_test)
    # print(X_test, y_test)
    return(X, y, X_train, X_test, y_train, y_test, sc)

# Create the ANN
def createANN(X_train, X_test, y_train, y_test, batchS, nEpochs):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from math import ceil
    import numpy as np
    # size of tests
    [rowX, colX] = np.shape(X_train)
    [row, col] = np.shape(y_train)
    unit = ceil((colX + col)/2)
    optim, lossT = 'nadam', 'mean_squared_error'
    classifier = Sequential()
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = colX)) # relu relu sigmoid
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = col, kernel_initializer = 'uniform', activation = 'linear'))      # sigmoid sigmoid linear
    classifier.compile(optimizer = optim, loss = lossT, metrics=['mae', 'acc'])
    classifier.fit(X_train, y_train, batch_size = batchS, epochs = nEpochs)
    # Predicting the Test set results
    scores = classifier.evaluate(X_test, y_test)
    print('PARAMETERS --> dense units: {0}, optimizer: {1} loss: {2} batch_size: {3} epochs: {4}'
        .format(unit, optim, lossT, batchS, nEpochs))
    print("{0}: {1} --> {2}: {3} %"
        .format(classifier.metrics_names[1], scores[1], classifier.metrics_names[2], scores[2]*100))
    return(classifier, scores, unit, optim, lossT)


def plotData(y_pred, y):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].set_title("Q1")
    axs[0,0].plot(y[:, 0], color = 'red', label = 'Test')
    axs[0,0].plot(y_pred[:, 0], color = 'blue', label = 'Predicted')
    axs[1,0].set_title("Q2")
    axs[1,0].plot(y[:, 1], color = 'red', label = 'Test')
    axs[1,0].plot(y_pred[:, 1], color = 'blue', label = 'Predicted')
    axs[0,1].set_title("Q3")
    axs[0,1].plot(y[:, 2], color = 'red', label = 'Test')
    axs[0,1].plot(y_pred[:, 2], color = 'blue', label = 'Predicted')
    try:
        axs[1,1].set_title("Q4")
        axs[1,1].plot(y[:, 3], color = 'red', label = 'Test')
        axs[1,1].plot(y_pred[:, 3], color = 'blue', label = 'Predicted')
    except (IndexError):
        pass
    plt.show()

def main():
    [input, target] = usrInput.getDataset()
    [X, y, X_train, X_test, y_train, y_test, sc] = importAndPrepare(input, target)
    batchS, nEpochs = 32, 100
    [classifier, scores, unit, optim, lossT] = createANN(X_train, X_test, y_train, y_test, batchS, nEpochs)
    y_pred = classifier.predict(X)
    plotData(y_pred, y)

if __name__ == "__main__":
    main()
