

import dataBaseClass

db = dataBaseClass.dataBase()
dataset = db.loadWSFunc(6)
[input, target] = db.dataToRNN(dataset, 4)
print(input)
