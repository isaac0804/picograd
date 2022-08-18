from picograd.tensor import Tensor

def main():
    
    a = Tensor(3.0)
    b = Tensor(4.0)

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