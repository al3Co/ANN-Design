# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

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

# Part 1 - Data Preprocessing
def importAndPrepare():
	# Importing the dataset
	dataset = pd.read_csv('Churn_Modelling.csv')
	X = dataset.iloc[:, 3:13].values
	y = dataset.iloc[:, 13].values

	# Encoding categorical data
	labelencoder_X_1 = LabelEncoder()
	X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
	labelencoder_X_2 = LabelEncoder()
	X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

	onehotencoder = OneHotEncoder(categorical_features = [1])
	X = onehotencoder.fit_transform(X).toarray()
	X = X[:, 1:]

	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	# Feature Scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	return(X_train, X_test, y_train, y_test, sc)



# Part 2 - Now let's make the ANN!
def createANN(X_train, X_test, y_train):

	# Initialising the ANN
	classifier = Sequential()

	# Adding the input layer and the first hidden layer
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dropout(rate = 0.1))

	# Adding the second hidden layer
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dropout(rate = 0.1))

	# Adding the output layer
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

	# Compiling the ANN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	# Fitting the ANN to the Training set
	classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)		# if y_pred > 0.5: value = true if not: value = false
	print(y_pred)
	return(classifier, y_pred)



def homework(classifier, sc, y_test, y_pred):
	# Predicting a single new observation
	"""Predict if the customer with the following informations will leave the bank:
	Geography: France
	Credit Score: 600
	Gender: Male
	Age: 40
	Tenure: 3
	Balance: 60000
	Number of Products: 2
	Has Credit Card: Yes
	Is Active Member: Yes
	Estimated Salary: 50000"""

	# Feature Scaling
	singleObservation = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
	singleObservation = sc.transform(singleObservation)
	new_prediction = classifier.predict(singleObservation)
	new_prediction = (new_prediction > 0.5)
	print(new_prediction)

	# Making the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN

def build_classifier1():
	classifier = Sequential()
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier

def evaluatingANN(X_train, y_train):
	classifier = KerasClassifier(build_fn = build_classifier1, batch_size = 10, epochs = 100)
	accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
	mean = accuracies.mean()
	variance = accuracies.std()
	print(mean)
	print(variance)

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN

def build_classifier2(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def tunningANN(X_train, y_train):
	classifier = KerasClassifier(build_fn = build_classifier2)
	parameters = {'batch_size': [25, 32],
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

# Part 5 - Now let's make the best ANN!
def createANNbest(X_train, X_test, y_train):
	classifier = Sequential()
	# Adding the input layer and the first hidden layer
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
	#classifier.add(Dropout(rate = 0.1))
	# Adding the second hidden layer
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
	#classifier.add(Dropout(rate = 0.1))
	# Adding the output layer
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	# Compiling the ANN
	classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Fitting the ANN to the Training set
	classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
	# Part 3 - Making predictions and evaluating the model
	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)
	print(y_pred)
	return(classifier, y_pred)


if __name__ == "__main__":
	X_train, X_test, y_train, y_test, sc = importAndPrepare()
	# classifier, y_pred = createANN(X_train, X_test, y_train)
	classifier, y_pred =createANNbest(X_train, X_test, y_train)
	homework(classifier, sc, y_test, y_pred)
	# evaluatingANN(X_train, y_train)
	#tunningANN(X_train, y_train)	# this takes long time