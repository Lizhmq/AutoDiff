class Node:
    _g = None
    def __init__(self):
        pass


class Constant(Node):
    count = 0
    def __init__(self, value, name=None):
        Node._g.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1

    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self):
        raise ValueError("Cannot assign values for Constant Node")


class Variable(Node):
    count = 0
    def __init__(self, value, name=None):
        Node._g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"


class Operator(Node):
    def __init__(self, name="Operator"):
        Node._g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name

    def forward(self):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError

    def __repr__(self):
        return f"Operator: name:{self.name}"

class PlaceHolder(Node):
    pass