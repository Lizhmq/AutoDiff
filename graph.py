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

    def 


with Graph() as g:
    x = Variable(1.3)
    y = Variable(0.9)
    z = x * y + 5

    print(g.variables)
    print(g.constants)
    print(g.operators)