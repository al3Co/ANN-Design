
# Importing the libraries
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

# Importing sklearn tools for prepare data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Importing the Keras libraries and packages
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
from scripts import usrInput

def importAndPrepare(X, y):
    y0 = y[:,[0]]
    y1 = y[:,[1]]
    y2 = y[:,[2]]
    try:
        y3 = y[:,[3]]
    except IndexError:
        y3 = 0
    return(X, y0, y1, y2, y3)

def build_classifier(optimizer):
    global colX, unit
    classifier = Sequential()
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'relu', input_dim = colX))
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    return classifier

def classifierParameters(X_train, y_train):
    global colX, unit
    [rowX, colX] = np.shape(X_train)
    unit = ceil((colX + 1)/2)
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [25, 32],
                  'epochs': [2, 5],
                  'optimizer': ['adam', 'rmsprop']}

    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(best_parameters)
    print(best_accuracy)

def main():
    [input, target] = usrInput.getDataset() # ValueError: continuous is not supported
    [X, y0, y1, y2, y3] = importAndPrepare(input, target)
    classifierParameters(X, y0)

if __name__ == "__main__":
    main()
