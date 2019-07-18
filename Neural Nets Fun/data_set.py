
from numpy import random



class DataSet(object):

    def __init__(self, X, Y, data_name):
        self.name = data_name
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        self.partitionTestSet(.2)

    def partitionTestSet(self, portion):
        X = self.X
        Y = self.Y
        n_samples = X.shape[0]
        n_test_samples = int(portion * n_samples)

        indexes = random.permutation(n_samples)
        indexes_test = indexes[:n_test_samples]
        indexes_train = indexes[n_test_samples:]
        self.X_test = X[indexes_test, :]
        self.Y_test = Y[indexes_test]
        self.X_train = X[indexes_train, :]
        self.Y_train = Y[indexes_train]

    
