import copy
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from tqdm.keras import TqdmCallback


class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        """
        Code from: https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
        Based on paper: Cyclical Learning Rates for Training Neural Networks
        https://arxiv.org/abs/1506.01186
        :param model:
        :param stopFactor:
        :param beta:
        """

        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []
        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
                       "DataFrameIterator", "Iterator", "Sequence"]
        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        from keras import backend as K

        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)
        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss
        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return
        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth
        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, trainData, startLR, endLR, epochs=None,
             stepsPerEpoch=None, batchSize=32, sampleSize=2048,
             verbose=1):
        # reset our class-specific variables
        self.reset()
        # determine if we are using a data generator or not
        useGen = self.is_data_iter(trainData)
        # if we're using a generator and the steps per epoch is not
        # supplied, raise an error
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)
        # if we're not using a generator then our entire dataset must
        # already be in memory
        elif not useGen:
            # grab the number of samples in the training data and
            # then derive the number of steps per epoch
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))
        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))
        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch
        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        from keras.callbacks import LambdaCallback
        from keras import backend as K
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)
        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
        self.on_batch_end(batch, logs))
        # check to see if we are using a data iterator
        if useGen:
            self.model.fit(
                x=trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                callbacks=[callback, TqdmCallback(verbose=0)],
                verbose=0)
        # otherwise, our entire training data is already in memory
        else:
            # train our model using Keras' fit method
            self.model.fit(
                x=trainData[0], y=trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                callbacks=[callback, TqdmCallback(verbose=0)],
                verbose=0)

        # restore the original model weights and set the optimal learning rate
        optim_lr = np.array(self.lrs)[np.argmin(self.losses)] / 10
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, optim_lr)
        return optim_lr

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]
        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)


class SkKerasBase:
    def __init__(self, epochs=1000, restarts=1,
                 validation_size=0.2, batch_size=None, criterion='mse', optimizer='Adam', lr=None,
                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100):
        super().__init__()
        self.epochs = epochs
        self.restarts = restarts

        # optimizer params
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.lr_lower_limit = lr_lower_limit
        self.lr_upper_limit = lr_upper_limit
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.n_epochs_without_improvement = n_epochs_without_improvement

    def define_architecture(self, X, y):
        raise Exception("Not implemented.")

    def fit_model(self, X, y, class_weight=None):
        self.batch_size = 1 if self.batch_size is None else self.batch_size
        self.batch_size = int(np.ceil(X.shape[0] * self.batch_size)) if self.batch_size <= 1 else self.batch_size

        if self.optimizer.lower() == "adam":
            import tensorflow as tf
            opt = tf.keras.optimizers.Adam
        else:
            try:
                from keras import optimizers
                opt = getattr(optimizers, self.optimizer)
            except:
                from tensorflow.keras import optimizers
                opt = getattr(optimizers, self.optimizer)

        # find learning rate
        if self.lr is None:
            model = self.define_architecture(X, y)
            model.compile(loss=self.criterion,
                          optimizer=opt(learning_rate=self.lr_lower_limit),
                          metrics=[])
            lrf = LearningRateFinder(model)
            self.lr = lrf.find([X, y], startLR=self.lr_lower_limit, endLR=self.lr_upper_limit,
                               stepsPerEpoch=np.ceil((len(X) / float(self.batch_size))),
                               batchSize=self.batch_size)
            # plot the loss for the various learning rates and save the
            # resulting plot to disk
            lrf.plot_loss()
            plt.show()
            del lrf
            # plt.savefig(config.LRFIND_PLOT_PATH)
            # self.find_lr(query, target)

        models = []
        for restart in range(self.restarts):
            from keras.callbacks import EarlyStopping  # , ModelCheckpoint
            # from keras.models import load_model

            print("Restart number: {}".format(restart))
            model = self.define_architecture(X, y)
            model.compile(loss=self.criterion,
                          optimizer=opt(learning_rate=self.lr),
                          metrics=[])
            # for other callbacks: https://keras.io/api/callbacks/#earlystopping
            history = model.fit(X, y,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_split=self.validation_size,
                                callbacks=[
                                    EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                  restore_best_weights=True,
                                                  patience=self.n_epochs_without_improvement),
                                    TqdmCallback(verbose=0)
                                ],
                                verbose=0,
                                class_weight=class_weight
                                )

            # with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            #     # for other callbacks: https://keras.io/api/callbacks/#earlystopping
            #     mc = ModelCheckpoint(fd.name, monitor='val_loss', mode='min', save_best_only=True)
            #     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
            #                        patience=self.n_epochs_without_improvement)
            #
            #     history = model.fit(query, target, epochs=self.epochs,
            #                         batch_size=self.batch_size, validation_split=self.validation_size,
            #                         callbacks=[es, mc, TqdmCallback(verbose=0)], verbose=0)
            #
            #     model = load_model(fd.name)
            models.append([copy.deepcopy(model), copy.deepcopy(history)])

        # models = list(map(train_in_paralel, range(self.restarts)))
        print("Models min validation loss: ", list(map(lambda x: min(x[1].history["val_loss"]), models)))
        model, _ = min(models, key=lambda x: min(x[1].history["val_loss"]))
        return model


class SkKerasMLP(SkKerasBase):
    def __init__(self, hidden_layer_sizes, activation, epochs=1000, restarts=1, validation_size=0.2, batch_size=None,
                 criterion='mse', optimizer='Adam', train_noise=0,
                 lr=None, lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.train_noise = train_noise
        SkKerasBase.__init__(self, epochs=epochs, restarts=restarts, validation_size=validation_size,
                             batch_size=batch_size, criterion=criterion, optimizer=optimizer, lr=lr,
                             lr_lower_limit=lr_lower_limit, lr_upper_limit=lr_upper_limit,
                             n_epochs_without_improvement=n_epochs_without_improvement)

    def define_architecture(self, query, target):
        from keras.layers import Dense, GaussianNoise
        from keras.models import Sequential

        model = Sequential()
        if self.train_noise > 0:
            model.add(GaussianNoise(self.train_noise, input_shape=np.shape(query)[1:]))
        model.add(Dense(self.hidden_layer_sizes[0], input_shape=np.shape(query)[1:], activation=self.activation))
        for hidden_layer_size in self.hidden_layer_sizes[1:]:
            model.add(Dense(hidden_layer_size, activation=self.activation))
        # model.add(Dense(np.prod(target.shape[1:]), activation='softmax'))
        return model

    def fit_model(self, X, y, class_weight=None):
        model = super().fit_model(X, y, class_weight=class_weight)
        weights, biases = zip(
            *[layer.get_weights() for layer in model.layers if len(layer.get_weights()) == 2])
        return weights, biases


class SkKerasClassifier(SkKerasMLP, MLPClassifier):
    def __init__(self, hidden_layer_sizes, activation, epochs=1000, restarts=1, validation_size=0.2, batch_size=None,
                 criterion='binary_crossentropy', optimizer='Adam', train_noise=0, class_weight=None,
                 lr=None, lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100):
        super(MLPClassifier, self).__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=1,
            solver='adam', alpha=0.0001, loss='log_loss',
            batch_size='auto', learning_rate="constant",
            learning_rate_init=0.001, power_t=0.5,
            shuffle=True, random_state=None, tol=1e-4,
            verbose=False, warm_start=False, momentum=0.9,
            nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
            epsilon=1e-8, n_iter_no_change=10, max_fun=15000
        )
        SkKerasMLP.__init__(self,
                            hidden_layer_sizes=hidden_layer_sizes, train_noise=train_noise,
                            activation=activation, epochs=epochs, restarts=restarts, validation_size=validation_size,
                            batch_size=batch_size, criterion=criterion, optimizer=optimizer, lr=lr,
                            lr_lower_limit=lr_lower_limit, lr_upper_limit=lr_upper_limit,
                            n_epochs_without_improvement=n_epochs_without_improvement)
        self.class_weight = class_weight

    def define_architecture(self, query, target):
        from keras.layers import Dense
        model = super().define_architecture(query, target)
        model.add(Dense(np.prod(target.shape[1:]), activation='softmax'))
        return model

    def fit(self, X, y):
        weights, biases = self.fit_model(X, y, class_weight=self.class_weight)
        super().fit(X, y)
        self.coefs_ = weights
        self.intercepts_ = biases
        return self


class SkKerasRegressor(SkKerasMLP, MLPRegressor):
    def __init__(self, hidden_layer_sizes, activation, epochs=1000, restarts=1, validation_size=0.2, batch_size=None,
                 criterion='mse', optimizer='Adam', train_noise=0,
                 lr=None, lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100):
        super(MLPRegressor, self).__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=1,
            solver='adam', alpha=0.0001, loss='log_loss',
            batch_size='auto', learning_rate="constant",
            learning_rate_init=0.001, power_t=0.5,
            shuffle=True, random_state=None, tol=1e-4,
            verbose=False, warm_start=False, momentum=0.9,
            nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
            epsilon=1e-8, n_iter_no_change=10, max_fun=15000
        )
        SkKerasMLP.__init__(self,
                            hidden_layer_sizes=hidden_layer_sizes, train_noise=train_noise,
                            activation=activation, epochs=epochs, restarts=restarts, validation_size=validation_size,
                            batch_size=batch_size, criterion=criterion, optimizer=optimizer, lr=lr,
                            lr_lower_limit=lr_lower_limit, lr_upper_limit=lr_upper_limit,
                            n_epochs_without_improvement=n_epochs_without_improvement)

    def define_architecture(self, query, target):
        from keras.layers import Dense
        model = super().define_architecture(query, target)
        model.add(Dense(np.prod(target.shape[1:])))
        return model

    def fit(self, X, y):
        weights, biases = self.fit_model(X, y)
        super().fit(X, y)
        self.coefs_ = weights
        self.intercepts_ = biases
        return self
