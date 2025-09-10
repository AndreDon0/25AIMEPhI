import numpy as np

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

a = np.array(matrix)
print(len(np.unique(a, axis=0)))