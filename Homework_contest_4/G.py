import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.tensor(list(map(float, input().split())), device=device)
view = list(map(int, input().split()))

print(tensor.view(view).to("cpu"))