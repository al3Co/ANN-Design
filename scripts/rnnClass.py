# rnn class

class RNNclass:
    unit = 32

    def __init__(self, batchS, nEpochs, optim_lossT, option):
        self.batchS = batchS
        self.nEpochs = nEpochs
        self.optimizer = optim_lossT[0]
        self.loss = optim_lossT[1]
        self.option = option            # to identify where the command comes

    def RNN(self, X, y):
        # Prepare data
        import numpy as np
        training_set = np.append(y, X, axis = 1)
        # Feature Scaling
        if self.option == 'reader':
            # Feature Scaling
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler(feature_range=(0.2, 0.8))
            training_set_scaled = sc.fit_transform(training_set)
        else:
            training_set_scaled = training_set
        # Creating a data structure with X timesteps and 1 output (can be other number, based on experience)
        X_train = []
        y_train = []
        timeSteps = 2
        numSamples = len(training_set_scaled)
        numData = len(training_set_scaled[0])
        nInputs = len(X[0])
        nOutputs = len(y[0])
        for nums in range(numData):
            for i in range(timeSteps, numSamples):
                X_train.append(training_set_scaled[(i-timeSteps):i, nums])
                y_train.append(training_set_scaled[i, nums])

        X_train, y_train = np.array(X_train), np.array(y_train)
        # Reshaping
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialising the RNN
        regressor = Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = self.unit, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        #regressor.add(Dropout(0.2))
        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = self.unit, return_sequences = True))
        #regressor.add(Dropout(0.2))
        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = self.unit, return_sequences = True))
        #regressor.add(Dropout(0.2))
        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = self.unit))
        #regressor.add(Dropout(0.2))
        # Adding the output layer
        regressor.add(Dense(units = 1))
        # Compiling the RNN
        regressor.compile(optimizer = self.optimizer, loss = self.loss, metrics=['mae', 'acc'])
        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = self.nEpochs, batch_size = self.batchS)

        scores = regressor.evaluate(X_train, y_train)
        print('PARAMETERS --> dense units: {0}, optimizer: {1} loss: {2} batch_size: {3} epochs: {4}'
            .format(self.unit, self.optimizer, self.loss, self.batchS, self.nEpochs))
        print("{0}: {1:.4f} --> {2}: {3:.4f} %"
            .format(regressor.metrics_names[1], scores[1], regressor.metrics_names[2], scores[2]*100))
        X_test = X_train
        y_pred = regressor.predict(X_test)
        return(regressor, scores[1], scores[2], self.unit)

    def RNN_save(self, model, name):
        # serialize model to JSON
        model_json = model.to_json()
        with open('results/models/' + name + '_Rmodel.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('results/models/' + name + '_Rmodel.h5')
        print("Saved model to disk")
