# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:05:10 2018

@author: Aldo Contreras
"""

import pandas as pd
from scripts import dataBaseClass

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
                    '[6] PCA vs IMUq\nSelect[]: ')
            kindOfTest = int(input(prompt))
            if (kindOfTest >= 1) and (kindOfTest <= 6):
                break
        except ValueError:
            print('ValueError')
    return kindOfTest

def getDataset():
    db = dataBaseClass.dataBase()
    kindOfData = kindOfDataFnc()
    kindOfTest = kindOfTestFnc()
    dataset = db.loadWSFunc(kindOfData)
    [input, target] = db.dataToRNN(dataset, kindOfTest)
    return(input, target)

if __name__ == "__main__":
    [input, target] = getDataset()
