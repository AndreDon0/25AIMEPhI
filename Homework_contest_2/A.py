import numpy as np

Height, Width = map(int, input().split())

i, j = np.indices((Height, Width))
array = np.where((i + j) % 2 == 0, 0, 255)

print(array)