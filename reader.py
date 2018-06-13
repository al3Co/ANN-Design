
# method to run ANNs and RNNs and save the results given in the csv file
import time
import csv
from scripts import dataBaseClass
import datetime

# specific parameters of the phenomenon
movDict = {'all':0,'combo':1,'cruzext':2,'cruzint':3,'elefront':4,'lateral':5,'rotz':6}

sortMov = {'FlexSvsShoulderAng':1, 'FlexSIMUqvsShoulderAng':2,
            'IMUqvsShoulderAng':3, 'PCAvsShoulder':4,
            'FlexSvsIMUq':5, 'PCAvsIMUq':6}

headers = ['Movement','Kind','Units','BatchSize',
            'Epochs','mae','acc','Time','FileName','Network']

option = 'reader'
# DL parameters
batch_size = [10, 20, 32]       # 10 20 32
numEpochs  = [100, 300, 500]      # 300 400
optim_lossT = ['nadam', 'mean_squared_error']
activ = ['sigmoid', 'linear']
network = {'ANN':0, 'RNN':1}

# function to run DL using dataset from data floder
def reader(optKey, optVal):
    if optVal == 0: from scripts import annClass
    else: from scripts import rnnClass
    db = dataBaseClass.dataBase()
    with open(('results/results'+ optKey + str(datetime.datetime.now())+'.csv'),'w') as resF:
        writer = csv.writer(resF, delimiter=',',lineterminator='\n',)
        writer.writerow(headers)
        for movK, movV in movDict.items():
            dataset = db.loadWSFunc(movV)
            for sortK, sortV in sortMov.items():
                [X, y] = db.dataToRNN(dataset, sortV)
                for nEpochs in numEpochs:
                    for batch in batch_size:
                        t = time.time()
                        print('Global Progress: ', optKey, batch, nEpochs, movK, sortK)
                        fileName = str(optKey + '_' + str(batch) + '_' + str(nEpochs) + movK + sortK)
                        if optVal == 0:
                            ann = annClass.ANNclass(batch, nEpochs, optim_lossT, activ, option)
                            [model, mae, acc, units] = ann.ANN(X,y)
                            ann.ANN_save(model, fileName)
                        else:
                            rnn = rnnClass.RNNclass(batch, (int(nEpochs/10)), optim_lossT, option)
                            [model, mae, acc, units] = rnn.RNN(X,y)
                            rnn.RNN_save(model, fileName)
                        writer.writerow([movK, sortK, units, batch, nEpochs, mae, acc, (time.time() - t), fileName, optKey])

# main
if __name__ == "__main__":
    for k, v in network.items():
        reader(k,v)
