import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = list(map(int, sys.stdin.read().split()))
N, M = data[0], data[1]
matrix1 = torch.tensor(data[2: N*M+2], device=device).view(N, M)
K, L = data[N*M+2], data[N*M+3]
matrix2 = torch.tensor(data[N*M+4:], device=device).view(K, L)

if M == K:
    print(int((matrix1.cpu() @ matrix2.cpu()).sum()))
else:
    print(int((matrix1.cpu() @ matrix2.mT.cpu()).sum()))
