import numpy as np

class NeuralNet(object):


    def __init__(self, W1, W2, activation, bias):
        self.W1 = W1 # first weight matrix
        self.W2 = W2 # second weight matrix
        self.activation = activation # activation function (usually step)
        self.bias = bias 
        if self.activation == 'step': # sets activation function to the right function
            self.activation_function = self.step
        elif self.activation == 'sigmoid':
            self.activation_function = self.sigmoid
        else:
            raise Exception('The given activation function is not a known function')


    def step(self, a):
        stepped_a = np.zeros(np.shape(a))
        if a.ndim == 1:
            for x_dim in range(np.shape(a)[0]):
                if a[x_dim] <= 0:
                    stepped_a[x_dim] = 0
                elif a[x_dim] > 0:
                    stepped_a[x_dim] = 1
        elif a.ndim == 2:
            for y_dim in range(np.shape(a)[0]):
                for x_dim in range(np.shape(a)[1]):
                    if a[y_dim][x_dim] <= 0:
                        stepped_a[y_dim][x_dim] = 0
                    elif a[y_dim][x_dim] > 0:
                        stepped_a[y_dim][x_dim] = 1
        return stepped_a


    def sigmoid(self, a):
        sigmoid_a = np.zeros(np.shape(a))
        if a.ndim == 1:
            for x_dim in range(np.shape(a)[0]):
                sigmoid_a = 1/(1+(np.exp(a[x_dim])))
        elif a.ndim == 2:
            for y_dim in range(np.shape(a)[0]):
                for x_dim in range(np.shape(a)[1]):
                    sigmoid_a = 1 / (1 + (np.exp(a[y_dim][x_dim])))
        return sigmoid_a


    def compute(self, nodes):
        self.hidden_nodes = np.ones((np.shape(nodes)[0]+1, ))
        self.hidden_nodes[:-1] = np.multiply(nodes)
        self.hidden_nodes[-1:] = self.bias



W1 = np.array([[1,1],
               [1,1],
               [-1,0]])

W2 = np.array([[-1],
                [1],
                [-1]])

xor = NeuralNet(W1,W2,'step')


nodes = np.array([[1,0,1]])


print(xor.compute(nodes))
