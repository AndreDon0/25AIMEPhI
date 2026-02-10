import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = list(map(int, sys.stdin.read().split()))
N, M = data[0], data[1]
limit = data[-1]
vals = torch.tensor(data[2:-1], device=device).view(N, M)

print(vals[vals > limit].sum().to("cpu"))
