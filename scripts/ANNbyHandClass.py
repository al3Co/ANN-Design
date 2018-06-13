import numpy as np
import time
from tqdm import tqdm

class ANNbyHand:
    lr = 0.1

    def __init__(self, epoch):
        self.epoch = epoch

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def derivatives_sigmoid(self, x):
        return x * (1 - x)

    def ANN(self, X, y):
        # Variable initialization
        # number of features in data set
        inputlayer_neurons = X.shape[1]
        # number of hidden layers neurons
        hiddenlayer_neurons = int((X.shape[1] + y.shape[1])/2) + 30
        # number of neurons at output layer
        output_neurons = y.shape[1]
        print('input leyer: {0} hidden leyer: {1} output leyer: {2}'
        .format(inputlayer_neurons, hiddenlayer_neurons, output_neurons))
        # weight and bias initialization
        wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
        bh=np.random.uniform(size=(1,hiddenlayer_neurons))
        wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
        bout=np.random.uniform(size=(1,output_neurons))

        for i in tqdm(range(self.epoch)):
            # Forward Propogation
            hidden_layer_input1=np.dot(X,wh)
            hidden_layer_input=hidden_layer_input1 + bh
            hiddenlayer_activations = self.sigmoid(hidden_layer_input)
            output_layer_input1=np.dot(hiddenlayer_activations,wout)
            output_layer_input= output_layer_input1+ bout
            output = self.sigmoid(output_layer_input)

            # Backpropagation
            E = y-output
            slope_output_layer = self.derivatives_sigmoid(output)
            slope_hidden_layer = self.derivatives_sigmoid(hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(wout.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            wout += hiddenlayer_activations.T.dot(d_output) * self.lr
            bout += np.sum(d_output, axis=0,keepdims=True) * self.lr
            wh += X.T.dot(d_hiddenlayer) * self.lr
            bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * self.lr

        return(output)
