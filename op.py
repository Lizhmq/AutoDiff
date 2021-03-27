from node import Node, Operator, Constant
import numpy as np
import math


class add(Operator):
    count = 0
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'add/{add.count}' if name is None else name
        add.count += 1
    
    def forward(self):
        a, b = self.inputs
        return a.value + b.value

    def backward(self, dout):
        return dout, dout


class minus(Operator):
    count = 0
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'minus/{minus.count}' if name is None else name
        minus.count += 1
    
    def forward(self):
        a, b = self.inputs
        return a.value - b.value

    def backward(self, dout):
        return dout, -dout


class multiply(Operator):
    count = 0
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'mul/{multiply.count}' if name is None else name
        multiply.count += 1
    
    def forward(self):
        a, b = self.inputs
        return a.value * b.value
    
    def backward(self, dout):
        a, b = self.inputs
        return dout * b.value, dout * a.value


class divide(Operator):
    count = 0
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'div/{divide.count}' if name is None else name
        divide.count += 1
    
    def forward(self):
        a, b = self.inputs
        return a.value / b.value
    
    def backward(self, dout):
        a, b = self.inputs
        return dout / b.value, dout * (- a.value / b.value / b.value)

class cos(Operator):
    count = 0
    def __init__(self, a, name=None):
        super.__init__(name)
        self.inputs = [a]
        self.name = f'cos/{cos.count}' if name is None else name
        cos.count += 1
    
    def forward(self):
        a = self.inputs[0]
        return math.cos(a.value)

    def backward(self, dout):
        a = self.inputs[0]
        return -math.sin(a.value) * dout


class sin(Operator):
    count = 0
    def __init__(self, a, name=None):
        super.__init__(name)
        self.inputs = [a]
        self.name = f'sin/{sin.count}' if name is None else name
        sin.count += 1
    
    def forward(self):
        a = self.inputs[0]
        return math.sin(a.value)

    def backward(self, dout):
        a = self.inputs[0]
        return math.cos(a.value) * dout


class tan(Operator):
    count = 0
    def __init__(self, a, name=None):
        super.__init__(name)
        self.inputs = [a]
        self.name = f'tan/{tan.count}' if name is None else name
        tan.count += 1
    
    def forward(self):
        a = self.inputs[0]
        return math.sin(a.value) / math.cos(a.value)

    def backward(self, dout):
        a = self.inputs[0]
        return dout / math.pow(math.cos(a.value), 2)


class exp(Operator):
    count = 0
    def __init__(self, a, name=None):
        super.__init__(name)
        self.inputs = [a]
        self.name = f'exp/{exp.count}' if name is None else name
        exp.count += 1
    
    def forward(self):
        a = self.inputs[0]
        return math.exp(a.value)

    def backward(self, dout):
        a = self.inputs[0]
        return dout * math.exp(a.value)


class log(Operator):
    count = 0
    def __init__(self, a, name=None):
        super.__init__(name)
        self.inputs = [a]
        self.name = f'log/{log.count}' if name is None else name
        log.count += 1
    
    def forward(self):
        a = self.inputs[0]
        return math.log(a.value)

    def backward(self, dout):
        a = self.inputs[0]
        return dout / a.value


def node_wrapper(func, self, other):
    """ Check to make sure that the two things we're comparing are
    actually graph nodes. Also, if we use a constant, automatically
    make a Constant node for it"""
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError("Incompatible types.")

Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__div__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1))
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)


class matmul(Operator):
    pass
    # count = 0
    # """Binary multiplication operation."""
    # def __init__(self, a, b, name=None):
    #     super().__init__(name)
    #     assert(isinstance(a, np.array))
    #     assert(isinstance(b, np.array))
    #     self.inputs = [a, b]
    #     self.name = f'matmul/{matmul.count}' if name is None else name
    #     matmul.count += 1
        
    # def forward(self):
    #     a, b = self.inputs
    #     return a@b
    
    # def backward(self, dout):
    #     a, b = self.inputs
    #     return dout@b.T, a.T@dout