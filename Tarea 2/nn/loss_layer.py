from nn.funcs import *
from nn.op import *
import numpy as np

#implements a log loss layer
class loss_layer(op):

    def __init__(self, i_size, o_size, func_acti, func_acti_grad, func_loss, func_loss_grad):
        super(loss_layer, self).__init__(i_size, o_size)
        self.func_acti = func_acti
        self.func_loss = func_loss
        if func_acti == softmax:
            self.func_backward = crossEntropySoftmax
        else:
            self.func_backward = lambda x, y : func_acti_grad(np.dot(self.x, self.W) + self.b) * func_loss_grad(self.o, y)  


    def forward(self, x):
        self.x = x
        self.o = self.func_acti(np.dot(x, self.W) + self.b)
        return self.o

    #alpha is used as reward in some reinforcement learning envs
    def backward(self, y, rewards=None):
        
        if self.func_acti == softmax:
            one_hot = np.zeros(self.o.shape)
            one_hot[np.arange(self.o.shape[0]), y] = 1
        else:
            one_hot = y

        if rewards is not None:
            self.grads = self.func_backward(self.o, one_hot) * rewards
        else:
            self.grads = self.func_backward(self.o, one_hot)
    def loss(self, y):
        if self.func_acti == softmax:
            one_hot = np.zeros(self.o.shape, dtype=np.int)
            one_hot[np.arange(self.o.shape[0]), y] = 1
        else:
            one_hot = y
        
        #fixed_section = np.nan_to_num((1 - one_hot) * np.log(1 - self.o))
        return self.func_loss(self.o, one_hot)
