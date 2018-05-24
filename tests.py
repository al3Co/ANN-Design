
# method to run ANNs and RNNs and save the results given in the csv file
import time
import csv
import dataBaseClass
import datetime

# specific parameters of the phenomenon
movDict = {'all':0,'combo':1,'cruzext':2,'cruzint':3,'elefront':4,'lateral':5,'rotz':6}
sortMov = {'FlexS vs ShoulderAng':1, 'FlexS+IMUq vs ShoulderAng':2,
            'IMUq vs ShoulderAng':3, 'PCA vs Shoulder':4,
            'FlexS vs IMUq':5, 'PCA vs IMUq':6}

# DL parameters
batch_size = [10, 20, 32]   # 10 20 32
numEpochs  = [300, 400]     # 300 400
optim_lossT = ['nadam', 'mean_squared_error']
activ = ['sigmoid', 'linear']
network = {'ANN':0, 'RNN':1}

# function to run DL using dataset from data floder
def reader(optKey, optVal):
    count = 0
    for movK, movV in movDict.items():
        for sortK, sortV in sortMov.items():
            for nEpochs in numEpochs:
                for batch in batch_size:
                    count += 1
                    print('Global Progress: ', optKey, batch, nEpochs, movK, sortK)
    print('Num NN: {}'.format(count))
# main
if __name__ == "__main__":
    for k, v in network.items():
        reader(k,v)
