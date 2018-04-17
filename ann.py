import numpy as np
import pandas as pd
from tqdm import tqdm

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

dataset = pd.read_csv('myData.csv')
# Input dataset matrix where each row is a training example   
X = dataset.iloc[:, 12:22].values

# Output dataset matrix where each row is a training example          
y = dataset.iloc[:, 5:6].values

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((len(X[0]),len(X))) - 1   # First layer of weights, Synapse 0, connecting l0 to l1.
syn1 = 2*np.random.random((len(y),len(y[0]))) - 1    # Second layer of weights, Synapse 1, connecting l1 to l2.

for j in tqdm(range(60000)):

    # Feed forward through layers 0, 1, and 2
    l0 = X                          # First Layer of the Network, specified by the input data
    l1 = nonlin(np.dot(l0,syn0))    # Second Layer of the Network, otherwise known as the hidden layer
    l2 = nonlin(np.dot(l1,syn1))    # Final Layer of the Network, which is our hypothesis, and should approximate the correct answer as we train.

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


print("Output After Training:")
print(l2)