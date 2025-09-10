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

# Преобразуем в numpy array
matrix = np.array(matrix)

mask = (matrix == 0).all(axis=0)

print(sum(mask))