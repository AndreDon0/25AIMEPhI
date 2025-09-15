import numpy as np

data = np.array(list(map(float, input().split())))
geometric_mean = np.prod(data) ** (1 / len(data))
print(geometric_mean)