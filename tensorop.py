from numpy.core.fromnumeric import reshape
from node import Node
from op import Operator
import numpy as np
import math


class Tensor2D(Node):
    count = 0
    def __init__(self, value, name=None):
        Node._g.tensors.add(self)
        assert(isinstance(value, np.ndarray))
        self.value = value
        self.shape = value.shape
        self.gradient = None
        self.name = f"Tensor/{Tensor2D.count}" if name is None else name
        Tensor2D.count += 1

    def __repr__(self):
        return f"Tensor: name:{self.name}, value:{self.value}"


class Conv2D(Operator):
    count = 0
    def __init__(self, inputs, weight, name=None):
        self.gradient = None
        self.inputs = inputs
        assert(isinstance(weight, np.ndarray))
        self.weight = weight
        self.weight_grad = None     # to be implemented
        self.name = f"Conv2D/{Conv2D.count}" if name is None else name
        Conv2D.count += 1

    def __repr__(self):
        return f"Tensor: name:{self.name}, value:{self.weight}"
    
    def forward(self):
        input = self.inputs[0].value
        inputdim = input.shape[0]
        convdim = self.weight.shape[0]
        padded = np.zeros((inputdim + convdim - 1, inputdim + convdim - 1))     # same padding
        start = (convdim - 1) // 2
        padded[start:start+inputdim, start:start+inputdim] = input
        out = np.zeros((inputdim, inputdim))
        for i in range(inputdim):
            for j in range(inputdim):
                out[i, j] = np.sum(padded[i:i+convdim, j:j+convdim] * self.weight)
        return out
    
    def backward(self, dout):
        inputdim = dout.shape[0]
        convdim = self.weight.shape[0]
        start = convdim // 2        # ceil((convdim - 1) / 2)
        padded = np.zeros((inputdim + convdim - 1, inputdim + convdim - 1))
        padded[start:start+inputdim, start:start+inputdim] = dout
        reversed_weight = np.zeros((convdim, convdim))
        for i in range(convdim):
            for j in range(convdim):
                reversed_weight[i, j] = self.weight[convdim - 1 - i, convdim - 1 - j]
        out = np.zeros((inputdim, inputdim))
        for i in range(inputdim):
            for j in range(inputdim):
                out[i, j] = np.sum(padded[i:i+convdim, j:j+convdim] * reversed_weight)
        return out


class Relu(Operator):
    count = 0
    def __init__(self, inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.name = f'Relu/{Relu.count}' if name is None else name
        Relu.count += 1
    
    def forward(self):
        out = np.array(self.inputs[0].value)
        return (out > 0) * out

    def backward(self, dout):
        return (dout > 0) * dout


class TensorAdd(Operator):
    count = 0
    def __init__(self, inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.name = f'TensorAdd/{TensorAdd.count}' if name is None else name
        TensorAdd.count += 1
    
    def forward(self):
        a, b = self.inputs
        return a.value + b.value

    def backward(self, dout):
        return dout, dout


class Flatten(Operator):
    """
        Flatten square matrix to 1d array.
    """
    count = 0
    def __init__(self, inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.name = f'Flatten/{Flatten.count}' if name is None else name
        Flatten.count += 1
    
    def forward(self):
        a = self.inputs[0].value
        return a.flatten()

    def backward(self, dout):
        dim = int(math.sqrt(dout.shape[0]))
        return np.reshape(dout, (dim, dim))


class FullyConn(Operator):
    count = 0
    def __init__(self, inputs, weight, bias=0, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.weight = weight
        self.bias = bias
        self.name = f'FullyConn/{FullyConn.count}' if name is None else name
        FullyConn.count += 1
    
    def forward(self):
        return np.sum(self.inputs[0].value * self.weight) + self.bias

    def backward(self, dout):
        return self.weight * dout