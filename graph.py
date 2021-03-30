import numpy as np
import math

from numpy.core.numeric import full
from node import *
from op import *
from tensorop import *
import json
from visual import make_graph

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
        self.tensors = set()
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
    print(f"Loading file {path}, function: {dic['Function']}")
    nodes = dic["Nodes"]
    inputs = dic["Inputs"]
    outputs = dic["Output"]
    name_dic = {}
    for node in nodes:
        if node["type"] == "PlaceHolder":
            newnode = None
            pass
        elif node["type"] in ["Constant", "Variable", "Tensor2D"]:
            typ = node["type"]
            node_dic = {
                "Constant": Constant,
                "Variable": Variable,
                "Tensor2D": Tensor2D
            }
            cls = node_dic[typ]
            value = np.array(node["value"]).astype(float) if typ == "Tensor2D" else node["value"]
            newnode = cls(value, node["name"])
        elif node["type"] == "Conv2D":
            inputs = [name_dic[na] for na in node["inputs"]]
            newnode = Conv2D(inputs=inputs, weight=np.array(node["weight"]).astype(float), name=node["name"])
        elif node["type"] == "FullyConn":
            inputs = [name_dic[na] for na in node["inputs"]]
            newnode = FullyConn(inputs=inputs, weight=np.array(node["weight"]).astype(float), bias=node["bias"], name=node["name"])
        else:
            typ = node["type"]
            operator_dic = {
                "add": add,
                "sin": sin,
                "cos": cos,
                "tan": tan,
                "log": log,
                "exp": exp,
                "multiply": multiply,
                "Relu": Relu,
                "TensorAdd": TensorAdd,
                "Flatten": Flatten
            }
            assert(typ in operator_dic.keys())
            cls = operator_dic[typ]
            inputs = [name_dic[na] for na in node["inputs"]]
            newnode = cls(inputs=inputs, name=node["name"])
        name_dic[node["name"]] = newnode
    _g.innodes = [name_dic[k] for k in dic["Inputs"]]
    _g.outnode = name_dic[dic["Output"]]
    return name_dic


def testconv():
    with Graph() as g:
        inputs = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 9]]).astype(float)
        inputs = Tensor2D(inputs)
        conv = Conv2D(inputs=[inputs], weight=np.ones((2, 2)).astype(float))
        flat = Flatten(inputs=[conv])
        fullyconn = FullyConn(inputs=[flat], weight=np.ones(9).astype(float))
        g.outnode = fullyconn
        g.forward_pass()
        y1 = fullyconn.value
        g.backward_pass()
        print(fullyconn.gradient)
        print(flat.gradient)
        print(conv.gradient)
        print(inputs.gradient)
        direction = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 9]]).astype(float)
        inputs.value += 0.001 * direction
        g.forward_pass()
        y2 = fullyconn.value
        print(y1, y2, (y2 - y1) / 0.001)
        print(np.sum(inputs.gradient * direction))
        print()


def validate_grad():
    with Graph() as g:
        file = "func2.json"
        global name_dic
        name_dic = read_graph(file, g)

        # input: x
        feed_dict = {"x1": 0, "x2": 0, "x3": 1}
        for name in feed_dict:
            name_dic[name].value = feed_dict[name]
        g.forward_pass()
        graph = make_graph(g.ordering)
        graph.render("graph1")

        y1 = g.outnode.value
        g.backward_pass()
        gradients = [name_dic[na].gradient for na in feed_dict]
        
        direction = {"x1": 1, "x2": 2, "x3": 3}
        vec = [direction[na] for na in feed_dict]
        print(f"<\\Nabla f(x), v>\t: {np.sum(np.array(gradients) * np.array(vec))}")


        # input2: x + tv
        t = 0.0001
        for name in feed_dict:
            name_dic[name].value += t * direction[name]
        g.forward_pass()
        y2 = g.outnode.value
        print(f"[f(x+tv) - f(x)] / t\t: {(y2 - y1) / t}")


def validate_grad2():
    with Graph() as g:
        file = "func1.json"
        global name_dic
        name_dic = read_graph(file, g)

        g.forward_pass()
        graph = make_graph(g.ordering)
        graph.render("graph2")

        y1 = g.outnode.value
        g.backward_pass()
        gradients = name_dic["x"].gradient
        direction = np.random.random(name_dic["x"].value.shape)
        print(f"<\\Nabla f(x), v>\t: {np.sum(np.array(gradients) * np.array(direction))}")

        t = 0.0001
        name_dic["x"].value += t * direction
        g.forward_pass()
        y2 = g.outnode.value
        print(f"[f(x+tv) - f(x)] / t\t: {(y2 - y1) / t}")



if __name__ == "__main__":
    # testconv()
    validate_grad()
    print()
    validate_grad2()
