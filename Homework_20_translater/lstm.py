import torch
from torch import nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=-1)
        gates = self.gates(combined)

        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, initial_state=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        if initial_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h, c = initial_state

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(self.proj(h))

        logits = torch.stack(outputs, dim=1)
        return logits, (h, c)
