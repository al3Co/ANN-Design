
# method to do all ANNs & RNNs and save on a dictionary the results given
import csv
import dataBaseClass
import datetime

movDict = {'all':0,'combo':1,'cruzext':2,'cruzint':3,'elefront':4,'lateral':5,'rotz':6}
sortMov = {'FlexS vs ShoulderAng':1, 'FlexS+IMUq vs ShoulderAng':2,
            'IMUq vs ShoulderAng':3, 'PCA vs Shoulder':4,
            'FlexS vs IMUq':5, 'PCA vs IMUq':6}
headers = ['Movement','Kind','Sequential','units','optimizer','loss',
            'batch_size','epochs','loss_value-mean_absolute_error-acc']
batch_size = [10, 20, 32]
numEpochs  = [300, 400]

options = {'ANN':0, 'RNN':1}

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
                [input, target] = db.dataToRNN(dataset, sortV)
                for nEpochs in numEpochs:
                    for batch in batch_size:
                        if optVal == 0:
                            [X, y, X_train, X_test, y_train, y_test, sc] = ann_4Ys.importAndPrepare(input, target)
                            [classifier, scores, unit, optim, lossT] = ann_4Ys.createANN(X_train, X_test, y_train, y_test, batch, nEpochs)
                        else: [classifier, scores, unit, optim, lossT] = rnn_4Ys.createRNN(input, target, batch, nEpochs)
                        writer.writerow([movK, sortK, classifier, unit, optim, lossT, batch, nEpochs, scores])
                        print('Global Progress: ', optKey, batch, nEpochs, movK, sortK)

if __name__ == "__main__":
    for k, v in options.items():
        reader(k,v)
