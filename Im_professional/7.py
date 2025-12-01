import numpy as np

d = int(input())
x0 = np.array(list(map(float, input().split())))
x1 = np.array(list(map(float, input().split())))
W = np.array([list(map(float, input().split())) for _ in range(d)])

v = x1 - x0
W_x0 = W @ x0
W_v = W @ v

term1 = np.outer(W_x0 - v, x0)
term2 = np.outer(W_x0 - v, v) / 2
term3 = np.outer(W_v, x0) / 2
term4 = np.outer(W_v, v) / 3

gradient = 2 * (term1 + term2 + term3 + term4)

for row in gradient:
    print(' '.join(map(str, row)))