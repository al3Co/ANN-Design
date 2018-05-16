
import numpy as np
import pandas as pd

def getClassifierDataset():
    dataset = pd.read_csv('data/allMovements.csv')
    X_train = dataset.iloc[:, 20:35].values   # FlexS
    y_train = dataset.iloc[:, -1].values     # kindOfMov
    dataset = pd.read_csv('data/comboAll.csv')
    X_test = dataset.iloc[:, 20:35].values    # FlexS
    y_test = dataset.iloc[:, -1].values     # kindOfMov
    return(X_train, y_train, X_test, y_test)

def importAndPrepare(y_train):
    # Encoding categorical data
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_train_transf = lb.fit_transform(y_train)
    return(lb, y_train_transf)

def buildClassifierANN(X_train, y_train, X_test):
    from keras.models import Sequential
    from keras.layers import Dense
    from math import ceil
    # ANN parameters
    [rowX, colX] = np.shape(X_train)
    [rowy, coly] = np.shape(y_train)
    unit = ceil((colX + coly)/2)
    optim, lossT = 'adam', 'binary_crossentropy'
    batchS, nEpochs = 32, 100
    # Initialising the ANN
    classifierANN = Sequential()
    classifierANN.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'relu', input_dim = colX))
    classifierANN.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'relu'))
    classifierANN.add(Dense(units = coly, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifierANN.compile(optimizer = optim, loss = lossT, metrics=['mae', 'acc'])
    classifierANN.fit(X_train, y_train, batch_size = batchS, epochs = nEpochs)
    scores = classifierANN.evaluate(X_train, y_train)
    print('PARAMETERS --> dense units: {0}, optimizer: {1} loss: {2} batch_size: {3} epochs: {4}'
        .format(unit, optim, lossT, batchS, nEpochs))
    print("{0}: {1} --> {2}: {3} %"
        .format(classifierANN.metrics_names[1], scores[1], classifierANN.metrics_names[2], scores[2]*100))
    y_pred = classifierANN.predict(X_test)
    return(classifierANN, (y_pred > 0.5))

def buildClassifierRNN():
    print('')

def buildClassifierBoltzmann_Machines():
    print('')

def decoding_inverse_transform(lb, y_pred):
    return(lb.inverse_transform(y_pred, threshold=None))

def main():
    # to train get all movements data to test get combo data
    [X_train, y_train, X_test, y_test] = getClassifierDataset()
    # change values to binary combination
    [lb, y_train_transf] = importAndPrepare(y_train)
    # build ANN to classify data
    [classifierANN, y_pred] = buildClassifierANN(X_train, y_train_transf, X_test)
    # transform predictions
    output_pred = decoding_inverse_transform(lb, y_pred)
    # print(output_pred)
    # TODO: from y_test change values in csv document from "COMBO" to movement wrote down on notebook
    # print(y_test)

if __name__ == "__main__":
    main()