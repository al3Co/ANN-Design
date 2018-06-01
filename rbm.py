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
dataset = pd.read_csv('data/rotacionzAll.csv')
X_test = dataset.iloc[:, 20:35].values    # FlexS
y_test = dataset.iloc[:, -1].values       # kindOfMov
del dataset

# scaling data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# adding kindOfMov
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train_transf = lb.fit_transform(y_train)
y_test_transf = lb.transform(y_test)
X_train = np.concatenate([X_train, y_train_transf], axis=1)
X_test = np.concatenate([X_test, y_test_transf], axis=1)

# getting number of samples
nb_samples = len(X_train)

pdb.set_trace() # breakpoint

# Converting the data into Torch tensors
training_set =np.array(X_train)
test_set = np.array(X_test)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[1])
nh = len(training_set[0])
rbm = RBM(nv, nh)

pdb.set_trace() # breakpoint

# Training the RBM
nb_epoch = 10
batch_size = 100
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_samples - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.0
# pdb.set_trace() # breakpoint
for id_user in range(nb_samples):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
