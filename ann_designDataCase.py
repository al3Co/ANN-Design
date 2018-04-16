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
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Data set

dataset = pd.read_csv('myData.csv')
X = dataset.iloc[:, 12:22].values	# flex sensor dataset
y = dataset.iloc[:, 5:9].values		# IMU Quat

print(len(X[0]), len(y[0]))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
y_train = sc.fit_transform(y_train)
# ANN

classifier = Sequential()
classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
#classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae', 'acc'])
classifier.fit(X_train, y_train, batch_size = 32, epochs = 500)
# Predicting the Test set results
y_final = classifier.predict(X_test)
y_final = sc.inverse_transform(y_final)
#print(y_final)

plt.plot(y_test, color = 'red', label = 'Test')
plt.plot(y_final, color = 'blue', label = 'Predicted')
plt.title('Data Prediction')
plt.xlabel('Sample')
plt.ylabel('Data')
plt.legend()
plt.show()

# # single prediction
# singleObservation = np.array([[1.54, 2.08, 1.53, 1.7, 1.5, 1.14, 2.29, 1.8, 1.51]])
# singleObservation = sc.transform(singleObservation)
# new_prediction = classifier.predict(singleObservation)
# print(new_prediction)