import numpy as np
import math
from node import *
from op import *

class Graph():
    """ Computational graph class. 
    Initilizes a global variable _g that describes the graph.
    Each graph consists of a set of
        1. operators
        2. variables
        3. constants
        4. placeholders
    """
    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        self.ordering = None
        self.outnode = None
        global _g
        _g = self
        Node._g = _g
    
    def reset_counts(self, root):
        """ 
            clear the count of node type
        """
        if hasattr(root, "count"):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        global _g
        del _g
        self.reset_counts(Node)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, trace):
        self.reset_session()

    def topo_sort(self, head_node=None):
        vis = set()
        curordering = []
        def _dfs(node):
            if node in vis:
                return
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
            curordering.append(node)
        
        if head_node is None:
            for node in self.operators:
                _dfs(node)
        else:
            _dfs(head_node)
        self.ordering = curordering
        return

    def forward_pass(self, feed_dict={}):
        if self.ordering is None:
            self.topo_sort(head_node=self.outnode)
        for node in self.ordering:
            if isinstance(node, PlaceHolder):
                node.value = feed_dict[node.name]
            elif isinstance(node, Operator):
                node.value = node.forward()
        if self.outnode is None:
            self.outnode = self.ordering[-1]
        return self.outnode.value
    
    def backward_pass(self):
        vis = set()
        self.outnode.gradient = 1
        for node in reversed(self.ordering):
            if isinstance(node, Operator):
                inputs = node.inputs
                grads = node.backward(node.gradient)
                for inp, grad in zip(inputs, grads):
                    if inp not in vis:
                        inp.gradients = grad
                        vis.add(inp)
                    else:
                        inp.gradients += grad
        return


with Graph() as g:
    x = Variable(1.3)
    y = Variable(0.9)
    z = x * y + 5

    print(g.variables)
    print(g.constants)
    print(g.operators)