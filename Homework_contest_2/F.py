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

print(str(matrix - matrix.mean(axis=1).reshape(-1, 1)))