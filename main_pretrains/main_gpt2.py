import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from transformers import GPT2Model, GPT2Config

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

class GPT2SOH(nn.Module):
    def __init__(self, input_dim, gpt2_model_path, PPA = False, soft_prompt_len=10):
        super(GPT2SOH, self).__init__()
        self.input_linear = nn.Linear(input_dim, 768)
        self.pos_encoder = PositionalEncoding(768)

        # Load the pre-trained GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_path, output_attentions=True, output_hidden_states=True)
        
        self.output_linear = nn.Linear(768, input_dim)

        # Freeze GPT-2 parameters except for LayerNorm and positional encodings
        for name, param in self.gpt2.named_parameters():
            if not any(layer in name.lower() for layer in ['ln', 'wpe', 'wte']):
                param.requires_grad = False

        self.PPA = PPA 
        if self.PPA:
            self.soft_prompt_len = soft_prompt_len                  
            # Initialize soft prompt
            self.soft_prompt = nn.Parameter(torch.zeros(self.soft_prompt_len, 768))
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
            # Add the soft prompt to the beginning of the input sequence
            batch_size = src.size(0)
            soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            src_with_prompt = torch.cat([soft_prompt_expanded, src], dim=1)
            
            # Use GPT-2 model
            gpt2_outputs = self.gpt2(inputs_embeds=src_with_prompt)
            hidden_states = gpt2_outputs.last_hidden_state
            hidden_states = hidden_states[:, self.soft_prompt.size(0):, :] # Remove the soft prompt from the output
        else:
            gpt2_outputs = self.gpt2(inputs_embeds=src)
            hidden_states = gpt2_outputs.last_hidden_state

        # Apply output linear layer
        output = self.output_linear(hidden_states) # batch, seq_len, 1
        # Denormalize the output
        output = output * stdev
        output = output + means
        return output, hidden_states.mean(dim=1)* stdev.squeeze(1) + means.squeeze(1)

def mask_input(src, mask_ratio=0.15):
    """
    Randomly mask input sequence.
    Args:
        src: input sequence of shape [batch_size, seq_len, feature_dim]
        mask_ratio: percentage of the sequence to be masked
    """
    src = src.clone()
    batch_size, seq_len, feature_dim = src.size()
    mask = torch.rand(batch_size, seq_len) < mask_ratio
    src[mask] = 0
    return src

def main_pretrain(train_loader, test_loader, model, optimizer, device, mask_ratio=0.15, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        train_loss = 0
        test_loss = 0
        for i, batch in enumerate(train_loader):
            src = batch[0].to(device)
            tgt = src.clone()
            src = mask_input(src, mask_ratio).to(device)
            optimizer.zero_grad()
            output, _ = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for i, batch in  enumerate(test_loader):
            src = batch[0].to(device)
            tgt = src.clone()
            src = mask_input(src, mask_ratio).to(device)
            output, _ = model(src)
            loss = criterion(output, tgt)
            test_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction (train) Loss: {train_loss / len(train_loader)}, Reconstruction (test) Loss: {test_loss / len(test_loader)}")


