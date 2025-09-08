n = int(input())
W = list(map(int, input().split()))
b = int(input())
X = list(map(int, input().split()))

def RELU(x):
    if x > 0:
        return x
    else:
        return 0

class Neuron:
    from typing import List, Callable
    def __init__(self, W: List[int], b: int, f: Callable[[int], int]) -> None:
        self.W = W
        self.b = b
        self.f = f
    
    def activate(self, X):
        if len(X) != len(self.W):
            raise Exception("Weighs' lenght is not equal to lenght of inputs")
        
        return self.f(sum([w * x for x, w in zip(self.W, X)]) + b)


N = Neuron(W, b, RELU)
print(N.activate(X))