import sys
import numpy as np
import torch

raw_data = sys.stdin.read().strip()
rows = raw_data.count('\n') + 1
flat_array = np.fromstring(raw_data, sep=' ', dtype=np.int32)
cols = flat_array.size // rows

tensor = torch.from_numpy(flat_array).view(rows, cols)

print(tensor[-2:, -2:].sum())