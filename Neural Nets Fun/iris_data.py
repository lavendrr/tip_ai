import csv
import numpy as np
from data_set import *

__all__ = ['iris_data']

with open('iris.data', 'r') as data_file:
    reader = csv.reader(data_file, delimiter=',')
    X = []
    Y = []
    for row in reader:
        if len(row) > 0: # last row is empty
            X += [row[:-1]]
            Y += [row[-1]]

X = np.array(X).astype('float64')

X = (X - X.mean(axis=0))/X.std(axis=0) # Normalize data by z-score



Y = np.array(Y)

labels = np.unique(Y)

n_labels = len(labels)

N = Y.shape[0] # number of samples

y_new = np.zeros((N, n_labels))
for i_sample in range(N): # for each sample
    y = Y[i_sample]
    i_label = np.where(labels==y)[0][0]
    y_new[i_sample, i_label] = 1.

Y = y_new

iris_data = DataSet(X, Y, 'Iris')
