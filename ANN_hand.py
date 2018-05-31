
import pdb
# test
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/allMovements.csv')
Xnp = dataset.iloc[0:4, 20:35].values   # FlexS
ynp = dataset.iloc[0:4, 41:44].values
del dataset

Xnp = np.array(Xnp)    # Input
ynp = np.array(ynp)    # Output

# scaling data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(Xnp)
y = sc.fit_transform(ynp)

from ANNbyHandClass import ANNbyHand
epoch = 50000
ann = ANNbyHand(epoch)
output = ann.ANN(X,y)
pdb.set_trace()
outputInv = sc.inverse_transform(output)
E = (ynp - outputInv)
print('max error: {}'.format(np.amax(E)))
