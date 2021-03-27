import numpy as np
import math
from node import *
from op import *
import json

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
        self.innodes = []
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
                if not isinstance(grads, tuple):
                    grads = [grads]
                # print(inputs)
                # print(grads)
                for inp, grad in zip(inputs, grads):
                    if inp not in vis:
                        inp.gradient = grad
                        vis.add(inp)
                    else:
                        inp.gradient += grad
        return


def read_graph(path, _g):
    with open(path, "r") as f:
        dic = json.load(f)
    nodes = dic["Nodes"]
    inputs = dic["Inputs"]
    outputs = dic["Output"]
    name_dic = {}
    for node in nodes:
        if node["type"] == "PlaceHolder":
            newnode = None
            pass
        elif node["type"] == "Constant":
            newnode = Constant(node["value"], node["name"])
        elif node["type"] == "Variable":
            newnode = Variable(node["value"], node["name"])
        else:
            typ = node["type"]
            operator_dic = {
                "add": add,
                "sin": sin,
                "cos": cos,
                "tan": tan,
                "log": log,
                "exp": exp,
                "multiply": multiply
            }
            assert(typ in operator_dic.keys())
            cls = operator_dic[typ]
            inputs = [name_dic[na] for na in node["inputs"]]
            newnode = cls(inputs=inputs, name=node["name"])
        name_dic[node["name"]] = newnode
    _g.innodes = [name_dic[k] for k in dic["Inputs"]]
    _g.outnode = name_dic[dic["Output"]]
    return name_dic


def validate_grad():
    with Graph() as g:
        file = "AutoDiff/func2.json"
        global name_dic
        name_dic = read_graph(file, g)
        # print(g.variables)
        # print(g.constants)
        # print(g.operators)

        # input: x
        feed_dict = {"x1": 0, "x2": 0, "x3": 1}
        for name in feed_dict:
            name_dic[name].value = feed_dict[name]
        g.forward_pass()
        y1 = g.outnode.value
        g.backward_pass()
        gradients = [name_dic[na].gradient for na in feed_dict]
        
        direction = {"x1": 1, "x2": 2, "x3": 3}
        vec = [direction[na] for na in feed_dict]
        print(f"<\\Nabla f(x), v>: {np.sum(np.array(gradients) * np.array(vec))}")


        # input2: x + tv
        t = 0.0001
        for name in feed_dict:
            name_dic[name].value += t * direction[name]
        g.forward_pass()
        y2 = g.outnode.value
        print(f"[f(x+tv) - f(x)] / t: {(y2 - y1) / t}")




if __name__ == "__main__":
    validate_grad()
    # with Graph() as g:
    #     file = "func2.json"
    #     name_dic = read_graph(file, g)
    #     # print(g.variables)
    #     # print(g.constants)
    #     # print(g.operators)
    #     feed_dict = {"x1": 0, "x2": 0, "x3": 1}
    #     for name in feed_dict:
    #         name_dic[name].value = feed_dict[name]
    #     g.forward_pass()
    #     print(g.outnode.value)