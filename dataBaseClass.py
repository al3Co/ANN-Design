import pandas as pd
import numpy as np
class dataBase:
    # fileKind = '.csv'
    # def __init__(self, path):
    #     self.a = path

    def loadWSFunc(self, kindOfData):
        # [0]All, [1]COMBO, [2]CRUZEXT, [3]CRUZINT, [4]ELEFRONT, [5]LATERAL, [6]ROTZ
        try:
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
            else:
                raise ValueError('0 to 6 value accepted')
        except ValueError as e:
            print(e)
        return dataset

    def dataToRNN(self, dataset, kindOfTest):
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
        else:
            input = None
            target = None
        return (input, target)
