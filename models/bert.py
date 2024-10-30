from transformers import BertModel, BertTokenizer
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
    
class BertSOH(nn.Module):
    def __init__(self, input_dim, bert_model_path, PPA=False, soft_prompt_len=10):
        super(BertSOH, self).__init__()
        self.input_linear = nn.Linear(input_dim, 768)
        self.pos_encoder = PositionalEncoding(768)

        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.output_linear = nn.Linear(768, input_dim)

        # Freeze BERT parameters except for LayerNorm and positional encodings
        for name, param in self.bert.named_parameters():
            if not any(layer in name.lower() for layer in ['ln', 'wpe', 'wte']):
                param.requires_grad = False

        self.PPA = PPA  # Prefix Prompt Adaptation
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
        src = self.pos_encoder(src)  # Add positional encoding

        if self.PPA:
            # Add the soft prompt to the beginning of the input sequence
            batch_size = src.size(0)
            soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            src_with_prompt = torch.cat([soft_prompt_expanded, src], dim=1)
            
            # Use BERT model
            bert_outputs = self.bert(inputs_embeds=src_with_prompt)
            hidden_states = bert_outputs.last_hidden_state
            hidden_states = hidden_states[:, self.soft_prompt.size(0):, :]  # Remove the soft prompt from the output
        else:
            bert_outputs = self.bert(inputs_embeds=src)
            hidden_states = bert_outputs.last_hidden_state

        # Apply output linear layer
        output = self.output_linear(hidden_states)  # batch, seq_len, input_dim
        # Denormalize the output
        output = output * stdev
        output = output + means
        return output, hidden_states.mean(dim=1) * stdev.squeeze(1) + means.squeeze(1)
