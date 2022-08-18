from picograd.tensor import Tensor

def main():
    a = Tensor(1.0)
    b = Tensor(2.0)
    print(a)
    print(b)

    c = a+b
    print(c)

    d = a*b
    print(d)

if __name__ == '__main__':
    main()