import sys
import torch

raw_data = sys.stdin.read()

if raw_data.find("\n\n\n") != -1:
    tensor = torch.tensor([[[list(map(int, d1.split(' '))) for d1 in d2.split('\n')] for d2 in d3.split("\n\n")] for d3 in raw_data.split("\n\n\n")])
    print(tensor.permute(2, 3, 1, 0).tolist())



elif raw_data.find("\n\n") != -1:
    tensor = torch.tensor([[list(map(int, d1.split(' '))) for d1 in d2.split('\n')] for d2 in raw_data.split("\n\n")])
    print(tensor.permute(1, 2, 0).tolist())
else:
    tensor = torch.tensor([list(map(int, d1.split(' '))) for d1 in raw_data.split('\n')])
    print(tensor.permute(1, 0).tolist())