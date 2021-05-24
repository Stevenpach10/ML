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

    def forward(self, x):
        self.x = x
        self.o = np.dot(x, self.W) + self.b
        return self.o

    def backward(self, y, rewards=None):
        if rewards is not None:
            self.grads = 2 * (y - self.o) * rewards
        else:
            self.grads = 2 * (y - self.o)

    def loss(self, y):
        return np.mean(np.power(y - self.o, 2))
