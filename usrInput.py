# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:05:10 2018

@author: Aldo Contreras
"""

import pandas as pd

def kindOfDataFnc():
    while True:
        try:
            prompt = ('[0]All [1]COMBO [2]CRUZEXT [3]CRUZINT [4]ELEFRONT [5]LATERAL [6]ROTZ \nTest: ')
            kindOfData = int(input(prompt))
            if (kindOfData >= 0) and (kindOfData <= 6):
                break
        except ValueError:
            print('Select a number between 0 and 6')
    return kindOfData

def kindOfTestFnc():
    while True:
        try:
            prompt = ('Kind\n [1] FlexS vs ShoulderAng\n [2] FlexS+IMUq vs ShoulderAng\n '
                    '[3] IMUq vs ShoulderAng\n [4] PCA vs Shoulder\n [5] FlexS vs IMUq\n '
                    '[6] PCA vs IMUq\n Select: ')
            kindOfTest = int(input(prompt))
            if (kindOfTest >= 1) and (kindOfTest <= 6):
                break
        except ValueError:
            print('ValueError')
    return kindOfTest

def loadWSFunc(kindOfData):
    # [0]All, [1]COMBO, [2]CRUZEXT, [3]CRUZINT, [4]ELEFRONT, [5]LATERAL, [6]ROTZ
    if kindOfData == 1:
        dataset = pd.read_csv('data/comboAll.csv')
    elif kindOfData == 2:
        dataset = pd.read_csv('data/cruzadoextAll.csv')
    elif kindOfData == 3:
        dataset = pd.read_csv('data/cruzadointAll.csv')
    elif kindOfData == 4:
        dataset = pd.read_csv('data/frontalAll.csv')
    elif kindOfData == 5:
        dataset = pd.read_csv('data/lateralAll.csv')
    elif kindOfData == 6:
        dataset = pd.read_csv('data/rotacionzAll.csv')
    elif kindOfData == 0:
        dataset = pd.read_csv('data/allData.csv')
    return dataset

def dataToRNN(dataset, kindOfTest):
    # [1]FlexS vs ShoulderAng [2]FlexS+IMUq vs ShoulderAng [3]IMUq vs ShoulderAng
    # [4]PCA vs ShoulderAng [5] FlexS vs IMUq [6] PCA vs IMUq
    if kindOfTest == 1:     # reviewed
        input = dataset.iloc[:, 20:35].values   # FlexS
        target = dataset.iloc[:, 41:44].values  # ShoulderAng
    elif kindOfTest == 2:
        input1 = dataset.iloc[:, 20:35].values  # FlexS
        input2 = dataset.iloc[:, 13:17].values  # IMUq
        input = np.concatenate([input1, input2], axis=1)    # FlexS + IMUq
        target = dataset.iloc[:, 41:44].values  # ShoulderAng
    elif kindOfTest == 3:   # reviewed
        input = dataset.iloc[:, 13:17].values   # IMUq
        target = dataset.iloc[:, 41:44].values  # ShoulderAng
    elif kindOfTest == 4:
        input = dataset.iloc[:, 20:35].values   # TODO PCA analysis
        target = dataset.iloc[:, 41:44].values  # ShoulderAng
    elif kindOfTest == 5:   # reviewed
        input = dataset.iloc[:, 20:35].values   # FlexS
        target = dataset.iloc[:, 13:17].values  # IMUq
    elif kindOfTest == 6:   # reviewed
        input = dataset.iloc[:, 20:35].values   # TODO PCA analysis
        target = dataset.iloc[:, 13:17].values   # IMUq
    return (input, target)

def getDataset():
    kindOfData = kindOfDataFnc()
    kindOfTest = kindOfTestFnc()
    dataset = loadWSFunc(kindOfData)
    [input, target] = dataToRNN(dataset, kindOfTest)
    return(input, target)

if __name__ == "__main__":
    [input, target] = getDataset()
