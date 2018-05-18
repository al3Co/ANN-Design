
import time
import numpy as np
import pandas as pd
import datetime
import csv
# specific parameters of the phenomenon
headers = ['Sequential','units','optimizer','loss',
            'batch_size','epochs','loss_value-mean_absolute_error-acc', 'Time']
# DL parameters
batch_size = [10, 20, 32]
numEpochs  = [100, 300]
optim_lossT = ['adam', 'binary_crossentropy']
activ = ['relu', 'sigmoid']
# network = {'ANN':0, 'RNN':1, 'BM':2}
network = {'ANN':0, 'RNN':1}
option = 'classifier'

def getClassifierDataset():
    dataset = pd.read_csv('data/allMovements.csv')
    X_train = dataset.iloc[:, 20:35].values   # FlexS
    y_train = dataset.iloc[:, -1].values      # kindOfMov
    dataset = pd.read_csv('data/comboAll.csv')
    X_test = dataset.iloc[:, 20:35].values    # FlexS
    y_test = dataset.iloc[:, -1].values       # kindOfMov
    return(X_train, y_train, X_test, y_test)

def importAndPrepare(y_train):
    # Encoding categorical data
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_train_transf = lb.fit_transform(y_train)
    return(lb, y_train_transf)

def buildClassifierBoltzmann_Machines():
    print('')

def decoding_inverse_transform(lb, y_pred):
    return(lb.inverse_transform(y_pred, threshold=None))

def classifierFunc(optKey, optVal, X, y):
    if optVal == 0: import ann_4Ys
    elif optVal == 1: import rnn_4Ys
    with open(('results/resultsClassifier'+ optKey + str(datetime.datetime.now())+'.csv'),'w') as resF:
        writer = csv.writer(resF, delimiter=',',lineterminator='\n',)
        writer.writerow(headers)
        for nEpochs in numEpochs:
            for batch in batch_size:
                t = time.time()
                print('Global Progress: ', optKey, batch, nEpochs)
                if optVal == 0: [classifier, scores, unit] = ann_4Ys.createANN(X, y, batch, nEpochs, optim_lossT, activ, option)
                elif optVal == 1: [classifier, scores, unit] = rnn_4Ys.createRNN(X, y, batch, (int(nEpochs/10)), optim_lossT, option)
                elif optVal == 2: buildClassifierBoltzmann_Machines()
                writer.writerow([classifier, unit, optim_lossT[0], optim_lossT[1], batch, nEpochs, scores, (time.time() - t)])

def main():
    [X_train, y_train, X_test, y_test] = getClassifierDataset() # to train get all movements data to test get combo data
    [lb, y_train_transf] = importAndPrepare(y_train)            # change values to binary combination
    for k, v in network.items():
        classifierFunc(k, v, X_train, y_train_transf)
    # TODO: from y_test change values in csv document from "COMBO" to movement wrote down on notebook

if __name__ == "__main__":
    main()
