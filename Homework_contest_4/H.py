import sys
import numpy as np
from io import StringIO
import torch

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_data = sys.stdin.read()
array = np.loadtxt(StringIO(raw_data), dtype=np.double)
tensor = torch.from_numpy(array).to(device)

print(int(torch.argmax(torch.mean(tensor, 1, True))))