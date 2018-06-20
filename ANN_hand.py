
import pdb

import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('data/test06.csv')
# Xnp = dataset.iloc[0:4, 26:32].values   # FlexSens
# ynp = dataset.iloc[0:4, 16:20].values   # IMUs
Xnp = dataset.iloc[:, 26:32].values   # FlexSens
ynp = dataset.iloc[:, 16:20].values   # IMUs
del dataset
# pdb.set_trace()

Xnp = np.array(Xnp)    # Input
ynp = np.array(ynp)    # Output

# scaling data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(Xnp)
y = sc.fit_transform(ynp)

from scripts import ANNbyHandClass
epoch = 150000
ann = ANNbyHandClass.ANNbyHand(epoch)
output = ann.ANN(X,y)

outputInv = sc.inverse_transform(output)
E = (ynp - outputInv)
print('max error: {}'.format(np.amax(E)))
pdb.set_trace()
print('Done')
