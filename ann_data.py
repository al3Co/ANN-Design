
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing sklearn tools for prepare data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Data Preprocessing
def importAndPrepare():
	# Importing the dataset
	dataset = pd.read_csv('myData.csv')
	X = dataset.iloc[:, 12:22].values	# flex sensor dataset
	y = dataset.iloc[:, 5:9].values		# IMU Quat

	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

	# Feature Scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	# print(X_test, y_test)
	return(X_train, X_test, y_train, y_test, sc)
	#return(X, X, y, y, sc)



# Create the ANN
def createANN(X_train, X_test, y_train):
	classifier = Sequential()
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
	classifier.add(Dropout(rate = 0.1))
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dropout(rate = 0.1))
	classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mae', 'acc'])
	classifier.fit(X_train, y_train, batch_size = 10, epochs = 500)
	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	# print(y_pred)
	return(classifier, y_pred)


# Single prediction function
def singlePrediction(classifier, sc, y_test, y_pred):
	# Feature Scaling
	singleObservation = np.array([[1.54, 2.08, 1.53, 1.7, 1.5, 1.14, 2.29, 1.8, 1.51, 1.81]])
	singleObservation = sc.transform(singleObservation)
	new_prediction = classifier.predict(singleObservation)
	print(new_prediction)

	# Making the Confusion Matrix
	# cm = confusion_matrix(y_test, y_pred)
	# print(cm)

# Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
def build_classifier1():
	classifier = Sequential()
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier

def evaluatingANN(X_train, y_train):
	classifier = KerasClassifier(build_fn = build_classifier1, batch_size = 5, epochs = 500)
	accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
	mean = accuracies.mean()
	variance = accuracies.std()
	print(mean)
	print(variance)

# Tuning the ANN
def build_classifier2(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def tuningANN(X_train, y_train):
	classifier = KerasClassifier(build_fn = build_classifier2)
	parameters = {'batch_size': [10, 32],
	              'epochs': [100, 500],
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

# Best ANN using tunning data
def createANNbest(X_train, X_test, y_train):
	classifier = Sequential()
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
	classifier.add(Dropout(rate = 0.1))
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dropout(rate = 0.1))
	classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
	classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	print(y_pred)
	return(classifier, y_pred)


if __name__ == "__main__":
	X_train, X_test, y_train, y_test, sc = importAndPrepare()
	classifier, y_pred = createANN(X_train, X_test, y_train)
	# classifier, y_pred = createANNbest(X_train, X_test, y_train)
	singlePrediction(classifier, sc, y_test, y_pred)
	# evaluatingANN(X_train, y_train)
	# tuningANN(X_train, y_train)	# this takes long time