# Boltzmann Machines mine
import pdb
# Importing the libraries
import numpy as np
import pandas as pd
# import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
dataset = pd.read_csv('data/allMovements.csv')
X_train = dataset.iloc[:, 20:35].values   # FlexS
y_train = dataset.iloc[:, -1].values      # kindOfMov
dataset = pd.read_csv('data/comboAll.csv')
X_test = dataset.iloc[:, 20:35].values    # FlexS
y_test = dataset.iloc[:, -1].values       # kindOfMov

del dataset

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0.2, 0.8))
X_train = sc.fit_transform(X_train)

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train_transf = lb.fit_transform(y_train)

training_set =np.array(X_train)
test_set = np.array(y_train_transf, dtype = 'int')

pdb.set_trace() # breakpoint

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
print(nb_users)
print(nb_movies)
