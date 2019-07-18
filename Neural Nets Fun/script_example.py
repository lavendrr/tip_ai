import numpy as np
from numpy import random, argmax
from sequential import *
from layers import *
from iris_data import iris_data
import matplotlib.pyplot as plt


### Training Parameters ###
n_batches = 300

#print_every = 10 # prints and saves performance every this many batches.

batch_size = 10

LR = 10**-1


# Set up training data
data = iris_data
n_out = 3
N = data.X_train.shape[0]
V = data.X.shape[1]

### Create Network ###
nn = Sequential((V,))
nn.add(Dense(64))
nn.add(Relu())
nn.add(Dense(n_out))
nn.add(Sigmoid())
nn.add(SSE())
nn.compile()

# Train network
history_acc_test = []
history_err_test = []
history_acc_train = []
history_err_train = []
#print('Epoch \tTrain Loss \t\tTest Loss \t\t Training Accuracy \t Testing Accuracy')
for i_batch in range(n_batches):
    #i = random.randint(N, size=batch_size)
    X = data.X_train#[i]
    Y = data.Y_train#[i]
    nn.train_batch(X, Y, LR)
    if True: #i_batch%print_every == 0:
        # Testing Accuracy
        nn.forward(data.X_test, data.Y_test)
        error_test = nn.layers[-1].value / data.X_test.shape[0]
        Y_pred = nn.layers[-2].value
        accuracy_test = (argmax(data.Y_test, axis=1) == argmax(Y_pred, axis=1)).mean()
        history_acc_test += [accuracy_test]
        history_err_test += [error_test]
        

        # Training Accuracy
        nn.forward(data.X_train, data.Y_train)
        error_train = nn.layers[-1].value / data.X_train.shape[0]
        Y_pred = nn.layers[-2].value #/ X.shape[0]
        accuracy_train = (argmax(data.Y_train, axis=1) == argmax(Y_pred, axis=1)).mean()
        history_acc_train += [accuracy_train]
        history_err_train += [error_train]
        
        #print(i_batch, " \t", error_train, '\t', error_test, " \t", accuracy_train, '\t', accuracy_test)

# Plot training progress
plt.title('Iris Dataset')
plt.xlabel('Epochs')
plt.ylabel('SSE')
plt.plot(history_err_train)
plt.plot(history_err_test)
plt.legend(['Train Error', 'Test Error'])
plt.show()

