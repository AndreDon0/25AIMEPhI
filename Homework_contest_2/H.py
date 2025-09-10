import sys
import numpy as np


data = sys.stdin.read().strip().splitlines()
angle = int(data[0])
# Соединяем оставшиеся строки и парсим как numpy array
arr_str = "\n".join(data[1:])
arr = eval(arr_str)   # осторожно, но в этой задаче вход именно в таком формате
arr = np.array(arr, dtype=np.uint8)

# Поворот без обрезки (reshape=True расширяет холст)
rotated = rotate(arr, angle, reshape=True, mode='constant', cval=0)

# Приводим к целому типа (если появились float после интерполяции)
rotated = np.rint(rotated).astype(np.uint8)

print(rotated)

