import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = torch.tensor(list(map(float, input().split())), device=device)
weights = torch.tensor(list(map(float, input().split())), device=device)
bies = torch.tensor(list(map(float, input().split())), device=device)

print(torch.sigmoid(features.matmul(weights) + bies).to("cpu"))