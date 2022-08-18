
class Tensor:
    def __init__(self, data):
        self.data = data
        # TODO: Shape
        # self.shape = shape

    def __repr__(self):
        return f"Tensor with data {self.data}"

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __rmul__(self, other):
        return self * other
