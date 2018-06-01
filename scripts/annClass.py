# ann class

class ANNclass:
    # aditional var

    def __init__(self, batch, nEpochs, optim_lossT, activ, option):
        self.batch = batch
        self.nEpochs = nEpochs
        self.optimizer = optim_lossT[0]
        self.loss = optim_lossT[1]
        self.activ = activ
        self.option = option            # to identify where the command comes

    def ANN(self, X, y):
        # Data Preprocessing
        from sklearn.model_selection import train_test_split
        if self.option == 'reader':
            from sklearn.preprocessing import MinMaxScaler
            # Splitting the dataset into the Training set and Test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
            # Feature Scaling
            sc = MinMaxScaler(feature_range = (0.2, 0.8))
            X_train = sc.fit_transform(X_train)
            y_train = sc.fit_transform(y_train)
            X_test = sc.fit_transform(X_test)
            y_test = sc.fit_transform(y_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
        # Creating ANN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from math import ceil
        import numpy as np
        # Size of tests
        [rowX, colX] = np.shape(X_train)
        [row, col] = np.shape(y_train)
        units = ceil((colX + col)/2)
        classifier = Sequential()
        classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = self.activ[0], input_dim = colX)) # relu relu sigmoid
        # classifier.add(Dropout(rate = 0.1))
        classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = self.activ[0]))
        # classifier.add(Dropout(rate = 0.1))
        classifier.add(Dense(units = col, kernel_initializer = 'uniform', activation = self.activ[1]))      # sigmoid sigmoid linear
        classifier.compile(optimizer = self.optimizer, loss = self.loss, metrics=['mae', 'acc'])
        classifier.fit(X_train, y_train, batch_size = self.batch, epochs = self.nEpochs)
        # Predicting the Test set results
        scores = classifier.evaluate(X_test, y_test)
        print('PARAMETERS --> dense units: {0}, optimizer: {1} loss: {2} batch_size: {3} epochs: {4}'
            .format(units, self.optimizer, self.loss, self.batch, self.nEpochs))
        print("{0}: {1:.4f} --> {2}: {3:.4f} %"
            .format(classifier.metrics_names[1], scores[1], classifier.metrics_names[2], scores[2]*100))
        return(classifier, scores[1], scores[2]*100, units)

    def ANN_save(self, model, name):
        # serialize model to JSON
        model_json = model.to_json()
        with open('results/models/' + name + '_Amodel.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('results/models/' + name + '_Amodel.h5')
        print("Saved model to disk")
