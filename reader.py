
# method to run ANNs and RNNs and save the results given in the csv file
import csv
import dataBaseClass
import datetime

# specific parameters of the phenomenon
movDict = {'all':0,'combo':1,'cruzext':2,'cruzint':3,'elefront':4,'lateral':5,'rotz':6}
sortMov = {'FlexS vs ShoulderAng':1, 'FlexS+IMUq vs ShoulderAng':2,
            'IMUq vs ShoulderAng':3, 'PCA vs Shoulder':4,
            'FlexS vs IMUq':5, 'PCA vs IMUq':6}
headers = ['Movement','Kind','Sequential','units','optimizer','loss',
            'batch_size','epochs','loss_value-mean_absolute_error-acc']
option = 'reader'
# DL parameters
batch_size = [10, 20, 32]   # 10 20 32
numEpochs  = [300, 400]     # 300 400
optim_lossT = ['nadam', 'mean_squared_error']
activ = ['sigmoid', 'linear']
network = {'ANN':0, 'RNN':1}

# function to run DL using dataset from data floder
def reader(optKey, optVal):
    if optVal == 0: import ann_4Ys
    else: import rnn_4Ys
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
                        print('Global Progress: ', optKey, batch, nEpochs, movK, sortK)
                        if optVal == 0: [classifier, scores, unit] = ann_4Ys.createANN(X, y, batch, nEpochs, optim_lossT, activ, option)
                        else: [classifier, scores, unit] = rnn_4Ys.createRNN(X, y, batch, (nEpochs/10), optim_lossT, option)
                        writer.writerow([movK, sortK, classifier, unit, optim_lossT[0], optim_lossT[1], batch, nEpochs, scores])

# main
if __name__ == "__main__":
    for k, v in network.items():
        reader(k,v)
