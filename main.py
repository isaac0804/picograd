from picograd.tensor import Tensor

def main():
    
    a = Tensor(3.0)
    b = Tensor([5.0, 2.0])
    c = Tensor([[1.0, 3.0], [5.0, 2.0], [1.0, 2.0]])
    print(a.shape)
    print(b.shape)
    print(c.shape)

    exit()

    def forward(a, b):
        c = a+b
        d = a*b + c
        e = d**2
        L = e
        return L

    L = forward(a, b)
    L.backward()
    print(b.grad)

    def test_grad():
        h = 1e-6
        grad = (forward(a, b+h) - forward(a, b)) / h
        print(grad)
    
    test_grad()


if __name__ == '__main__':
    main()