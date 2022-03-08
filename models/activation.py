import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter

########################################################################
# Activation function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def none_func(x):
    return x

def get_act(activation, in_features=1):
    if activation=='relu':
        return lambda x: F.relu(x, inplace=True)
    elif activation=='relu6':
        return lambda x: F.relu6(x, inplace=True)
    elif activation=='soft_relu6':
        return soft_relu6
    elif activation=='soft_relu6_':
        return soft_relu6_
    elif activation=='tanh':
        return torch.tanh
    elif activation=='logsigmoid6':
        return logsigmoid6
    elif activation=='atan':
        return lambda x: 2*torch.atan(x)
    elif activation=='sin':
        return torch.sin
    elif activation=='srelu':
        return SReLU(in_features)
    elif activation=='hswish':
        return HSwish(in_features)
    elif activation=='swish':
        return Swish(in_features)
    elif activation=='I':
        return none_func
    else:
        print('Unsupported Activation %s'%(activation))
        # return activation
        return none_func

class SReLU(nn.Module):
    def __init__(self, in_features, parameters=None):
        """
        Init method.
        """
        super(SReLU, self).__init__()
        self.in_features = in_features

        # Init with leaky relu 0.01
        if parameters is None:
            self.tr = Parameter(
                torch.zeros((1,in_features, 1, 1), dtype=torch.float, requires_grad=True)
            )
            self.tl = Parameter(
                torch.zeros((1,in_features, 1, 1), dtype=torch.float, requires_grad=True)
            )
            self.ar = Parameter(
                torch.ones((1,in_features, 1, 1), dtype=torch.float, requires_grad=True)
            )
            self.al = Parameter(
                -0.01*torch.ones((1,in_features, 1, 1), dtype=torch.float, requires_grad=True)
            )
        else:
            self.tr, self.tl, self.ar, self.al = parameters

    def forward(self, input):
        """
        Forward pass of the function
        """
        return (
            (input >= self.tr).float() * (self.tr + self.ar * (input + self.tr))
            + (input < self.tr).float() * (input > self.tl).float() * input
            + (input <= self.tl).float() * (self.tl + self.al * (input + self.tl))
        )

class HSwish(nn.Module):
    def __init__(self, in_features=1, parameters=None):
        super(HSwish, self).__init__()
        if parameters is None:
            self.beta = Parameter(6.0*torch.ones((1,in_features, 1, 1), dtype=torch.float, requires_grad=True))
        else:
            self.beta = parameters
        self.zero = Parameter(torch.zeros((1,in_features, 1, 1), dtype=torch.float, requires_grad=False))

    def forward(self, input):
        return input.mul(input.add(self.beta/2).minimum(self.beta).maximum(self.zero).div(self.beta))

class Swish(nn.Module):
    def __init__(self, in_features=1, parameters=None):
        super(Swish, self).__init__()
        if parameters is None:
            self.beta = nn.Parameter(torch.ones((1,in_features, 1, 1), dtype=torch.float, requires_grad=True))
        else:
            self.beta = parameters

    def forward(self, input):
        return input.mul(input.mul(self.beta).sigmoid())

def logsigmoid6(x):
    return F.logsigmoid(x.relu().sub(6)).add(6)

def soft_relu6_(x):
    return x.sub(3.0).tanh().add(1).mul(3)

def soft_relu6(x):
    x_detach = x.detach()
    return soft_relu6_(x) + (F.relu6(x_detach).sub(soft_relu6_(x_detach)))