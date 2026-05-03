import torch
from torch import nn

import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=-1)

        z, r = self.gates(combined).chunk(2, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_candidate = torch.cat([x, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.candidate(combined_candidate))

        h_next = (1 - z) * h_prev + z * h_tilde

        return h_next


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, initial_state=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        if initial_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = initial_state

        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(self.proj(h))

        logits = torch.stack(outputs, dim=1)
        return logits, h
