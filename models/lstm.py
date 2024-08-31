import torch.nn as nn
import torch
import numpy as np

class LSTM_SOH(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, PPA=False, soft_prompt_len=10):
        super(LSTM_SOH, self).__init__()
        self.input_linear = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layers, batch_first=True)
        self.output_linear = nn.Linear(embed_dim, input_dim)
        self.PPA = PPA
        if self.PPA:
            self.soft_prompt_len = soft_prompt_len
            self.soft_prompt = nn.Parameter(torch.zeros(self.soft_prompt_len, embed_dim))
            nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

    def forward(self, src):
        means = src.mean(1, keepdim=True).detach()
        src = src - means
        stdev = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5)
        src /= stdev

        src = self.input_linear(src)
        if self.PPA:
            batch_size = src.size(0)
            soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            src = torch.cat([soft_prompt_expanded, src], dim=1)
            output, (hn, cn) = self.lstm(src)
            output = output[:, self.soft_prompt_len:, :]
            hn = hn[:, -batch_size:, :] 
        else:   
            output, (hn, cn) = self.lstm(src)

        output = self.output_linear(output)
        output = output * stdev
        output = output + means
        return output, hn[-1] * stdev.squeeze(1) + means.squeeze(1)
