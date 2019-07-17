import numpy as np

class NeuralNet(object):


    def __init__(self, num_input, num_hidden, num_output, activation):
        self.w1 = np.random.rand(num_input, num_hidden)
        self.w2 = np.random.rand(num_hidden, num_output)

        self.b1 = np.random.randint(0,100)
        self.b2 = np.random.randint(0,100)

        self.h1 = np.ones((num_hidden, ))
        self.h2 = np.ones((num_output, ))

        self.activation = activation

        if self.activation == 'step':
            self.activation_function = self.step
        elif self.activation == 'sigmoid':
            self.activation_function = self.sigmoid
        else:
            raise Exception('the given activation function is not a known function')


    def step(self, a):
        stepped_a = np.zeros(np.shape(a))

        if a.ndim == 1:
            for x_dim in range(np.shape(a)[0]):
                if a[x_dim] <= 0:
                    stepped_a[x_dim] = 0
                elif a[x_dim] > 0:
                    stepped_a[x_dim] = 1
            return stepped_a
        raise Exception('input is not a vector')


    def sigmoid(self, a):
        sigmoid_a = np.zeros(np.shape(a))

        if a.ndim == 1:
            for x_dim in range(np.shape(a)[0]):
                sigmoid_a = 1/(1+(np.exp(a[x_dim])))
            return sigmoid_a
        raise Exception('input is not a vector')


    def forward(self, h0):
        if np.shape(h0)[0] != self.num_input:
            raise Exception('wrong amount of nodes inputted')

        self.h1[:-1] = np.multiply(h0, self.w1)
        self.h1[-1:] = self.b1

        self.h2[:-1] = np.multiply(self.activation_function(self.h1), self.w2)
        self.h2[-1:] = self.b2

        return self.h2


xor = NeuralNet(3,3,1,'step')


print('W1 is \n \n {} \n \n W2 is \n \n {} \n \n The 1st bias is {} \n \n The 2nd bias is {}'.format(xor.w1, xor.w2, xor.b1, xor.b2))


print('\n \n ')
