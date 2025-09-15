import numpy as np

N = int(input())

i, j = np.indices((N + 1, N + 1))
arr = i * j

print(arr[1:, 1:])
