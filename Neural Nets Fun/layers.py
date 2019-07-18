import numpy as np
from numpy import zeros, ones, exp, concatenate, random, arange, unravel_index, repeat, ndindex

__all__ = ['Input', 'Sigmoid', 'Relu', 'SSE', 'Flatten', 'Dense', 'ZeroPad', 'SoftMax', 'LinearConvolve', 'MaxPool']

# The classes in the file are in the "Layer" abstract class.
# 
# layer.shape is output dim. does not include N, batch size.
# layer.value: output of function layer represents of shape N+layer.shape
# layer.sen: sensitivity; dL/d(value)
# layer.forward: Applies function to previous layer's value to get self.value
# layer.backward: finds sensitivity of previous layer.


class Input(object):
    def __init__(self, input_dims):
        self.shape = tuple(input_dims)
        self.is_trainable = False
    def forward(self):
        pass
    def backward(self):
        pass


class Sigmoid(object):
    """Takes in layer of any dimensions and has value of same size."""
    def init(self, in_layer):
        self.in_layer = in_layer
        self.shape = in_layer.shape
        self.is_trainable = False
        
    def forward(self):
        x = self.in_layer.value
        self.value = 1./(1. + exp(-x))

    def backward(self):
        y = self.value
        sen_y = self.sen
        x_sen = sen_y * (y * (1. - y))
        self.in_layer.sen = x_sen

class SSE(object):
    def __init__(self):
        self.shape = (1,)
        self.is_trainable = False
        
    def init(self, in_layer):
        self.in_layer = in_layer

    def forward(self, Y):
        Y_pred = self.in_layer.value # N, H1
        self.error = Y - Y_pred
        self.value = ((self.error)**2).sum()

    def backward(self):
        sen_x = -2 * self.error # N, H1
        self.in_layer.sen = sen_x
        

class Relu(object):
    """Takes in layer of any dimensions and has value of same size."""
    def init(self, in_layer):
        self.in_layer = in_layer
        self.shape = in_layer.shape
        self.is_trainable = False
        
    def forward(self):
        x = self.in_layer.value
        self.value = (x > 0.) * x
    
    def backward(self):
        x = self.in_layer.value
        sen_y = self.sen
        sen_x = (x > 0.)*sen_y
        self.in_layer.sen = sen_x


class Flatten(object):
    """Takes in layer of at least 2 dimensions and flattens the last 2."""
    def init(self, in_layer):
        self.in_layer = in_layer
        self.shape = (np.prod(in_layer.shape),)
        self.is_trainable = False

    def forward(self):
        x = self.in_layer.value
        self.value = x.reshape((-1, self.shape[0]))
        
    def backward(self):
        sen_y = self.sen
        sen_x = sen_y.reshape((-1,)+self.in_layer.shape)
        self.in_layer.sen = sen_x

class Dense(object):
    """
        Takes in layer of (N x H0).
        Outputs N x H1, By multiplication by an H0 x H1 matrix.
    """
    def __init__(self, H1):
        self.H1 = H1
        self.shape = (H1,)
        
    def init(self, in_layer):
        self.in_layer = in_layer
        H0 = in_layer.shape[0]
        self.W = (1./H0) * (random.random((H0, self.H1)) - .5)
        self.b = random.random(self.H1)
        self.params = [self.W, self.b]
        self.is_trainable = True
        
    def forward(self):
        X = self.in_layer.value # N x H0
        W = self.W              # H0 x H1
        b = self.b
        self.value = X.dot(W) + b   # N x H1
        
    def backward(self):
        sen_y = self.sen        # N x H1
        W = self.W              # H0 x H1
        sen_x = sen_y.dot(W.T)  # N x H0
        self.in_layer.sen = sen_x

    def gradient(self):
        N = self.sen.shape[0]
        sen_y = self.sen            # N x H1
        x = self.in_layer.value   # N x H0
      #  print(x.shape)
      #  print(sen_y.shape)
      #  print('fdsa')
        sen_W = (x.T).dot(sen_y) # H0 x H1
        sen_b = sen_y.sum(axis=0) # H1
        return [sen_W, sen_b]

    def updateParams(self, LR):
        N = self.value.shape[0]
        sen_W = self.gradient()[0]
        sen_b = self.gradient()[1]
        self.W -= (LR/N) * sen_W
        self.b -= (LR/N) * sen_b
        
        

class ZeroPad(object):
    """
        Takes in layer of shape (N, C, n_in, m_in).
        Surrounds last 2 dimensions with pad 0's on all sides.
    """
    def __init__(self, pad):
        self.pad = pad
        
    def init(self, in_layer):
        pad = self.pad
        self.in_layer = in_layer
        in_shape = in_layer.shape
        n_out = in_shape[-2]+2*pad
        m_out = in_shape[-1]+2*pad
        self.shape = in_shape[:-2] + (n_out, m_out)
        self.is_trainable = False

    def forward(self):
        pad = self.pad
        x = self.in_layer.value
        y = zeros((x.shape[0],)+self.shape)
        y[:, :, pad:-pad, pad:-pad] = x
        self.value = y
        
    def backward(self):
        pad = self.pad
        sen_y = self.sen
        sen_x = sen_y[:, :, pad:-pad, pad:-pad]
        self.in_layer.sen = sen_x

class SoftMax(object):
    """
        Takes an array of shape (N, H).
        Returns an array of shape (N, H).
    """
    def init(self, in_layer):
        self.in_layer = in_layer
        self.shape = in_layer.shape
        self.is_trainable = False
        
    def forward(self):
        x = self.in_layer.value # (N, H)
        m = x.max(axis=1, keepdims=True) # (N, 1)
        a = np.exp(x-m)         # (N, H)
        T = a.sum(axis=1, keepdims=True) # (N, 1)
        probs = a/T              # (N, H)
        self.value = probs

    def backward(self, y): # y: (N,)
        N = self.value.shape[0]
        probs = self.value # (N, H)
        #sen_prob = self.sen # N 1-hot vectors (N, H)
        #j = sen_prob.argmin(axis = 1) # (N, 1)
        sen_scores = np.copy(probs) # (N, H)
        
        sen_scores[arange(N), y] -= 1 #
        self.in_layer.sen = sen_scores

class Convolve(object):
    """
        In shape: (N, C, n_in, m_in)
    """
    def __init__(self, filter_shape, stride):
     #   self.filter_shape = (n_filters,)+filter_shape
      #  self.F = n_filters
        #self.F = n_filters
        self.n_f = filter_shape[0]
        self.m_f = filter_shape[1]
        self.stride = stride

    def init(self, in_layer):
        # F = specified by child
        stride = self.stride
        self.in_layer = in_layer
        F = self.F
        n_f     = self.n_f
        m_f     = self.m_f
        #print(in_layer.shape)
        C, n_in,  m_in  = in_layer.shape
        
        self.filter_shape = n_f, m_f
        n_out = int((n_in - n_f)/stride + 1); assert(n_out%1==0); n_out=int(n_out);
        m_out = int((m_in - m_f)/stride + 1); assert(m_out%1==0); m_out=int(m_out);
        self.shape = (F, n_out, m_out)
        
    def forward(self):
        F, n_out, m_out = self.shape
        n_f, m_f = self.filter_shape#[2:]
        X = self.in_layer.value # (N, C, n_in, m_in)
        N = X.shape[0]
        filter_func = self.filter_func
        #F, D, n_f, m_f = filter_func.filter_shape
        F = self.F
        stride = self.stride
        result = np.zeros((N, F, n_out, m_out))

        for i, j in np.ndindex(n_out, m_out): # for each output position
            ii = stride*i # input position
            jj = stride*j # input position
            x_i = X[:, :, ii:ii+n_f, jj:jj+m_f] # get slice for input to filter
            result[:, :, i, j] = filter_func.forward(x_i)
        if self.is_trainable:
            result += self.params[1]
        self.value = result

    def backward(self):
        in_layer = self.in_layer.value
        #sen_y = self.sen
        N = in_layer.shape[0]
        F, n_out, m_out = self.shape
        n_f, m_f = self.filter_shape#[2:]
        stride = self.stride
        sen_in_layer = zeros(in_layer.shape)
        for i, j in ndindex((n_out, m_out)): # for output dims
            ii = stride*i # input position
            jj = stride*j # input position
            x = in_layer[:, :, ii:ii+n_f, jj:jj+m_f] # get slice for input to filter
            sen_y = self.sen[:, :, i, j]
            sen_in_layer[:, :, ii:ii+n_f, jj:jj+m_f] += self.filter_func.backward(x, sen_y)
        self.in_layer.sen = sen_in_layer

    def gradient(self):
        filter_func = self.filter_func
        in_layer = self.in_layer.value
        stride = self.stride
        F, n_out, m_out = self.shape
        F2, C2, n_f, m_f = filter_func.filter_shape
        sen_W = zeros(filter_func.filter_shape)
        for i, j in ndindex((n_out, m_out)): # for output dims
            ii = stride*i # input position
            jj = stride*j # input position
            x = in_layer[:, :, ii:ii+n_f, jj:jj+m_f] # get slice for input to filter
            sen_y = self.sen[:, :, i, j]
            sen_W += filter_func.outer(x, sen_y)

        sen_b = self.sen.sum(axis=0)
        
        return [sen_W, sen_b]

    def updateParams(self, LR):
        N = self.value.shape[0]
        sen_W = self.gradient()[0]
        sen_b = self.gradient()[1]
        self.params[0] -= (LR/N) * sen_W
        self.params[1] -= (LR/N) * sen_b
                    
        
class LinearConvolve(Convolve):
    def __init__(self, n_filters, filter_size=(3,3), stride=1):
        Convolve.__init__(self, filter_size, stride)
        #self.n_f = filter_shape[0]
        #self.m_f = filter_shape[1]
        #self.stride = stride
        self.F = n_filters
        self.filter_size = filter_size
        
    def init(self, in_layer):
        Convolve.init(self, in_layer)
        C = in_layer.shape[0]
        F = self.F
        n_inputs = C * self.filter_shape[0] * self.filter_shape[1]
        W = (1./n_inputs) * (random.random((F, C) +self.filter_shape) - .5)
        b = random.random(self.shape) - .5
        self.params = [W, b]
        self.filter_func = LinearFilter(W)
        self.is_trainable = True

class MaxPool(Convolve):
    def __init__(self, pool_size = (2, 2), stride=2):
        Convolve.__init__(self, pool_size, stride)
        #self.n_f = filter_shape[0]
        #self.m_f = filter_shape[1]
        #self.stride = stride
        
    def init(self, in_layer):
        self.F = in_layer.shape[0] # in & out have same number of channels
        Convolve.init(self, in_layer)
        self.filter_func = MaxPoolFilter()
        self.is_trainable = False

class LinearFilter(object):
    """
        Represents a linear convolutional filter.
        Initialized with a (F, C, n_f, m_f) array
    """
    def __init__(self, W):
        self.filter_shape = W.shape
        self.W = W # (F, C, n_f, m_f)
        
    def forward(self, x):
        """
            x: (N, C, n_f, m_f) slice 
            Returns: (N, F).
        """
        W = self.W[None]    # (1, F, C, n_f, m_f)
        X = x[:, None]      # (N, 1, C, n_f, m_f)
        return (self.W * X).sum(axis=(2, 3, 4))

    def backward(self, x, sen_y):
        """
            x:       (N, C, n_f, m_f) slice
            sen_y:   (N, F)-array of output sensitivity.
            Returns: (N, C, n_f, m_f) slice array.
        """
        W = self.W[None]                        # (1, F, C, n_f, m_f)
        sen_y2 = sen_y[:, :, None, None, None]  # (N, F, 1, 1,   1)
        return (self.W * sen_y2).sum(axis=1)    # (N,    C, n_f, m_f)

    def outer(self, x, sen_y):
        """
            x:       (N, C, n_f, m_f)
            y:       (N, F)
            Returns: (F, C, n_f, m_f)
        """
        X = x[:, None]                          # (N, 1, C, n_f, m_f)
        sen_y2 = sen_y[:, :, None, None, None]  # (N, F, 1, 1,   1)
        return (X*sen_y2).sum(axis=0)           # (   F, C, n_f, m_f)
        
        
class MaxPoolFilter(object):
    def __init__(self):
        pass
        #self.filter_size = filter_size

    def forward(self, x):
        """
            x: (N, C, n_f, m_f)
            returns (N, C)
        """
        #sen_x = zeros(x.shape)
        N, C, n_f, m_f = x.shape
        xx = x.reshape(N, C, n_f*m_f)   # (N, C, ff)    # flatten the two spatial dimensions
        ii = xx.argmax(axis=2)          # (N, C)        # find max in that area
        i, j = unravel_index(ii, (n_f, m_f)) # (N, C), (N, C) # convert to original indexing
        k = repeat(arange(N)[:, None], C, axis=1) # (N, C)
        p = repeat(arange(C)[None, :], N, axis=0) # (N, C)
        y = x[k, p, i, j]
        return y

    def backward(self, x, sen_y):
        """
            x:       (N, C, n_f, m_f)
            sen_y:   (N, C)
            Returns: (N, C x n_f x m_f) slice.
        """
        sen_x = zeros(x.shape)
        N, C, n_f, m_f = x.shape
        xx = x.reshape(N, C, n_f*m_f)   # (N, C, ff)    # flatten the two spatial dimensions
        ii = xx.argmax(axis=2)          # (N, C)        # find max in that area
        i, j = unravel_index(ii, (n_f, m_f)) # (N, C), (N, C) # convert to original indexing
        k = repeat(arange(N)[:, None], C, axis=1) # (N, C)
        p = repeat(arange(C)[None, :], N, axis=0) # (N, C)
        sen_x[k, p, i, j] = sen_y
        return sen_x
        

class QuadForm(object):

    def __init__(self, H1):
        self.shape = (H1,)
        self.is_trainable = True

    def init(self, in_layer):
        self.in_layer = in_layer
        H0 = in_layer.shape[0]
        H1 = self.shape[0]
        W = (1./(H0+1)**2) * (random.random((H1, H0+1, H0+1)) - .5)
        self.params = [W]

    def forward(self):
        X = self.in_layer.value     # X: N, H0
        W = self.params[0]          # W: H1 ,H0+1, H0+1
        N = X.shape[0]
        #H1 = W.shape[0]
        X = concatenate([X, ones((N, 1))], axis=1) # N, H0+1
        xr = X[:, None, :, None]  # N, 1, H0+1, 1
        xc = X[:, None, None, :] # N, 1, 1, H0+1
        yy = xr * W * xc # N, H1, H0+1, H0+1
        y = yy.sum(axis=3).sum(axis=2)
        self.value = y

    def backward(self):
        # finds sen_x: (N, H0)
        X = self.in_layer.value # N, H0
        N = X.shape[0]
        X = concatenate([X, ones((N, 1))], axis=1) # N, H0+1
        sen_y = self.sen    # N, H1
        W = self.params[0] # H1 , H0+1, H0+1
        wi = W[None, :, :, :] * X[:, None, :, None] # mult x by rows of W
        wj = W[None, :, :, :] * X[:, None, None, :] # mult x by cols of W
        wii = wi.sum(axis=2) # sum over cols
        wjj = wj.sum(axis=3) # sum over rows # N, H1, H0+1
        
        sx = sen_y[:, :, None] * (wii + wjj) # N, H1, H0+1
        
        sen_x = sx.sum(axis=1) # N, H0+1
        
        self.in_layer.sen = sen_x[:, :-1]
        

    def gradient(self):
        X = self.in_layer.value # X: N, H0
        W = self.params[0] # W: H1 ,H0+1, H0+1
        sen_y = self.sen # N, H1
        N = X.shape[0]
        H1 = sen_y.shape[1]
        X = concatenate([X, ones((N, 1))], axis=1) # N, H0+1
        Xo = X[:, None, :] * X[:, :, None] # N, H0+1, H0+1    X outer
        sen_yo = sen_y[:, :, None, None] # N, H1, 1, 1
        sen_ww = Xo[:, None] * sen_yo # N, H1, H0+1, H0+1
        sen_W = sen_ww # N, H1, H0+1, H0+1
        return [sen_W.sum(axis=0)]

    def updateParams(self, LR):
        N = self.value.shape[0]
        sen_W = self.gradient()[0]
        self.params[0] -= (LR/N) * sen_W



