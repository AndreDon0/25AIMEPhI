from math import ceil
import numpy as np

# Считываем k
k = int(input())

# Считываем матрицу построчно
matrix = []
while True:
    try:
        line = input()
        if line.strip() == "":
            break
        row = list(map(int, line.split()))
        matrix.append(row)
    except EOFError:
        break

# Преобразуем в numpy array
matrix = np.array(matrix)
expanded_matrix = np.zeros((ceil(matrix.shape[0] / k) * k, ceil(matrix.shape[1] / k) * k), dtype=int)
expanded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
reshaped_matrix = expanded_matrix.reshape(expanded_matrix.shape[0]//k, k, expanded_matrix.shape[1]//k, k)
result = reshaped_matrix.sum(axis=(1, 3))

for row in result:
    print(' '.join(map(str, row)))