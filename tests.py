
# method to run ANNs and RNNs and save the results given in the csv file

from scripts import dataBaseClass

db = dataBaseClass.dataBase()
dataset = db.loadWSFunc(6)          # rotz = 6
[X, y] = db.dataToRNN(dataset, 1)   # FlexS vs ShoulderAng = 1

batch = 32   # 10 20 32
epochs  = 2    # 300 400
optim_lossT = ['adam', 'mean_squared_error']
activ = ['sigmoid', 'linear']
option = 'reader'

name = 'ANNtest'
nameR = 'RNNtest'

from scripts import annClass
ann = annClass.ANNclass(batch, epochs, optim_lossT, activ, option)
[model, mae, acc, units] = ann.ANN(X,y)
ann.ANN_save(model, name)

print(model, mae, acc, units)


from scripts import rnnClass
rnn = rnnClass.RNNclass(batch, epochs, optim_lossT, option)
[model, mae, acc, units] = rnn.RNN(X,y)
rnn.RNN_save(model, nameR)

print(model, mae, acc, units)
