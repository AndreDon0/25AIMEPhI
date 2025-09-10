import sys
import numpy as np


data = sys.stdin.read().strip()
arr = eval(data)
arr = np.array(arr, dtype=np.uint8)

arr = arr.reshape(-1, arr.shape[2])
unique_colors = np.unique(arr, axis=0)

print(unique_colors.shape[0])
for color in unique_colors:
    print(*color)

