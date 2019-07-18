import numpy as np
from numpy import log, ndindex, arange, copy
from layers import *


__all__ = ['Sequential']

class Sequential(object):

    def __init__(self, input_dims):
        self.layers = [Input(input_dims)]

    def add(self, layer):
        self.layers += [layer]

    def compile(self):
        in_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.init(in_layer)
            in_layer = layer

    def forward(self, value, Y_true=None):
        """Only executes loss layer if given value for 'Y_true'."""
        self.layers[0].value = value
        for layer in self.layers[1:-1]:
            layer.forward()
        if type(Y_true) != type(None):
            self.layers[-1].forward(Y_true)
        return self.layers[-2].value # prediction from final layer

    def backward(self):
        layers = self.layers
        layers[-1].backward()
        for layer in reversed(layers[:-1]):
            layer.backward()

    def updateParams(self, LR):
        for layer in self.layers:
            if layer.is_trainable:
                layer.updateParams(LR)

    def loss(self, X, Y):
        probs = self.forward(X)
        N = X.shape[0]
        total_loss = -log(probs[arange(N), Y])
        return total_loss.mean()

    def train_batch(self, X, Y, LR):
        self.forward(X, Y)
        self.backward()
        self.updateParams(LR)
    """
    def train(self, X, Y, LR, batch_size, n_epochs, print_every):
        N = X.shape[0]
        for epoch in range(n_epochs):
            i = random.randint(N, size=batch_size)
            X = X_train[i]
            Y = Y_train[i]
            self.train_batch(X, Y, LR)
    """
    def checkGradients(self, x, y, epsilon=.0001, tolerance=10**-5):
        layers = self.layers
        N = x.shape[0]
        self.forward(x, y)
        loss1 = copy(self.layers[-1].value) # get start loss for comparison
        self.backward()                # calculate gradients of intermediate values
        
        for i_layer in range(len(layers)-1):
            self.forward(x, y) # reset all intermediate values
            layer = self.layers[i_layer]
            print('\nLayer: ', layer)
            print('index \t calculated \t\t emperical \t\t percent error')
            A_s = [layer.value] # layer value
            Sen_s = [layer.sen] # sensitivity of that value
            if layer.is_trainable:
                A_s += layer.params
                Sen_s += layer.gradient()
            for i_a in range(len(A_s)):
                print (i_a)
                A = A_s[i_a]
                sen = Sen_s[i_a]
                for index in ndindex(A.shape):
                    A[index] += epsilon # perterb value
                    if i_a > 0: # if this is a trainable parameter
                        layer.forward() # run this layer
                    for L in self.layers[i_layer+1:-1]: # run subsequent layers
                        L.forward()
                    self.layers[-1].forward(y)
                    
                    
                    loss2 = self.layers[-1].value          # find new loss
                    d_loss = loss2 - loss1           # change in loss
                    emperical_grad = d_loss/epsilon # approximate emperical gradient
                    calc_grad = sen[index] 
                    percent_error = (calc_grad - emperical_grad)/emperical_grad
                    A[index] -= epsilon             # return weight to original value
                    try:
                        if (emperical_grad!=calc_grad) and (percent_error > tolerance):
                            print (str(index) + "\t" + str(calc_grad) + "\t" + str(emperical_grad) + "\t" + str(percent_error))
                    except:
                        print (index)
                        print( emperical_grad)
                        print(calc_grad)
                        print(sen.shape)
                        raise ValueError


        
        
                
            







