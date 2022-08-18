
class Tensor:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(_children)

        # TODO: Shape
        # self.shape = shape

    def backward(self):
        nodes = []

        def build_graph(x):
            if x not in nodes:
                for c in x._children:
                    build_graph(c)
                nodes.append(x)
        build_graph(self)

        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()
        
        for node in nodes:
            print(node, node.grad)
    
    def relu(self):
        out = Tensor(self.data * (self.data > 0), (self,))

        def _backward():
            """
            L = f(x)
            x = relu(a)
            dL/da = dL/dx * dx/da
            """
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor with data {self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            """
            L = f(x)
            x = a+b
            dL/da = dL/dx * dx/da = dL/dx
            dL/db = dL/dx * dx/db = dL/dx
            """
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            """
            L = f(x)
            x = a*b
            dL/da = dL/dx * dx/da = dL/dx * b
            dL/db = dL/dx * dx/db = dL/dx * a
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other))

        def _backward():
            # Assume b is a constant
            """
            L = f(x)
            x = a**b
            dL/da = dL/dx * b * a ** (b - 1)
            """
            self.grad += other.data * self.data ** (other.data - 1) * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other ** -1.0

    def __rtruediv__(self, other):  # other / self
        return (self / other) ** -1.0

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other