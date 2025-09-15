import numpy as np

N = int(input())

i, j = np.indices((N, N))

arr = (i + j) % 2 == 0

print(arr.astype(int))
