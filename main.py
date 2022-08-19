from picograd.tensor import Tensor

def main():
    
    a = Tensor([9.0, 1.1])
    b = Tensor([5.0, 2.0])
    print(a.shape)
    print(b.shape)

    def forward(a, b):
        c = a+b
        d = a*b + c
        e = d**2
        L = e
        return L

    L = forward(a, b)
    L.backward()
    print(a.grad)

    def test_grad():
        h = Tensor([1e-6, 1e-6])
        grad = (forward(a+h, b) - forward(a, b)) / h
        print(grad)
    
    test_grad()


if __name__ == '__main__':
    main()