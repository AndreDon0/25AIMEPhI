import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = int(input())
seqs = [torch.tensor(list(map(int, input().split())), device=device) for _ in range(n)]

# padded = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=0)
# print(padded)

m = max([tensor.size(0) for tensor in seqs])

answer = [F.pad(tensor, (0, m - tensor.size(0)), mode='constant', value=0).tolist() for tensor in seqs if m != tensor.size(0)]

for a in answer:
    print(*a)
