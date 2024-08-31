import torch.nn as nn
import torch
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerSOH(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, PPA = False, soft_prompt_len = 10):
        super(TransformerSOH, self).__init__()
        self.input_linear = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_linear = nn.Linear(embed_dim, input_dim)
        self.PPA = PPA # 是否使用Prefix Prompt Adaptation
        if self.PPA:
            self.soft_prompt_len = soft_prompt_len                  
            # Initialize soft prompt
            self.soft_prompt = nn.Parameter(torch.zeros(self.soft_prompt_len, embed_dim))
            nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

    def forward(self, src):
        # Instance normalization on src
        means = src.mean(1, keepdim=True).detach()
        src = src - means
        stdev = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5)
        src /= stdev

        src = self.input_linear(src)  # Apply linear transformation
        # Add positional encoding
        src = self.pos_encoder(src)

        if self.PPA:
            batch_size = src.size(0)
            soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            src_with_prompt = torch.cat([soft_prompt_expanded, src], dim=1)
            memory = self.transformer_encoder(src_with_prompt)
            # memory = memory[:, self.soft_prompt_len:, :]  # Remove the prompt part from the memory
        else:
            memory = self.transformer_encoder(src)
        
        # Use the memory for decoding
        if self.PPA:
            src_for_decoder = src_with_prompt
        else:
            src_for_decoder = src
        
        output = self.transformer_decoder(src_for_decoder, memory)
        output = self.output_linear(output)
        output = output[:, self.soft_prompt_len:, :] 
        
        # Denormalize the output
        output = output * stdev
        output = output + means
        return output, memory[:, self.soft_prompt_len:, :].mean(dim=1) * stdev.squeeze(1) + means.squeeze(1) # 输出重建的output和中间表示