
# method to do all ANNs & RNNs and save on a dictionary the results given
import csv
import dataBaseClass
import ann_4Ys
import datetime

movDict = {'all':0,'combo':1,'cruzext':2,'cruzint':3,'elefront':4,'lateral':5,'rotz':6}
sortMov = {'FlexS vs ShoulderAng':1, 'FlexS+IMUq vs ShoulderAng':2,
            'IMUq vs ShoulderAng':3, 'PCA vs Shoulder':4,
            'FlexS vs IMUq':5, 'PCA vs IMUq':6}
headers = ['Movement','Kind','classifier','units','optimizer','loss',
            'batch_size','epochs','loss_value-mean_absolute_error-acc']
batch_size = 32
numEpochs  = [100, 200]

def readerANN():
    db = dataBaseClass.dataBase()
    with open(('results/resultsANN'+str(datetime.datetime.now())+'.csv'),'w') as resF:
        writer = csv.writer(resF, delimiter=',',lineterminator='\n',)
        writer.writerow(headers)
        for movK, movV in movDict.items():
            dataset = db.loadWSFunc(movV)
            for sortK, sortV in sortMov.items():
                [input, target] = db.dataToRNN(dataset, sortV)
                for nEpochs in numEpochs:
                    [X, y, X_train, X_test, y_train, y_test, sc] = ann_4Ys.importAndPrepare(input, target)
                    [classifier, scores, unit, optim, lossT] = ann_4Ys.createANN(X_train, X_test, y_train, y_test, batch_size, nEpochs)
                    writer.writerow([movK, sortK, classifier, unit, optim, lossT, batch_size, nEpochs, scores])

def readerRNN():
    db = dataBaseClass.dataBase()



if __name__ == "__main__":
    reader()
