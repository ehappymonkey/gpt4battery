import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Tokenizer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False).float()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output
    
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class RBNet(nn.Module):
    def __init__(self, configs):
        super(RBNet, self).__init__()
        
        self.pred_len = configs.pred_len
        self.d_ff = configs.d_ff
        self.seq_len = configs.seq_len
        
        self.gpt2 = GPT2Model.from_pretrained('./gpt2', output_attentions=True, output_hidden_states=True)
    
        self.tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.gpt2.parameters():
            param.requires_grad = False
        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
        #         param.requires_grad = True
        #     elif 'mlp' in name and configs.mlp == 1:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        self.out_layer = nn.Linear(768, configs.c_out)
        
        self.value_embedding = TokenEmbedding(c_in=configs.enc_in, d_model=configs.d_model)
        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)

        self.mainhead = nn.Linear(52, 1) # main task head
        self.sshead = nn.Linear(52, 49)

        self.patch_len = 7 #
        self.stride = 1 #
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.word_embeddings = self.gpt2.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.d_llm = 768
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)


    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        
        # return np.squeeze(dec_out[:, -self.pred_len:, :])# , dec_out #batch=10, seq_len=50, 1       # hidden_states  # [B, L, D]
        return dec_out#, soh, recon# , hidden_states
        # dec_out = self.mainhead(dec_out.permute(0, 2, 1))
        # return dec_out


    def forecast(self, x_enc): # 输入维度是B, L, M
        B, L, M = x_enc.shape 
        x_enc = self.normalize_layers(x_enc, 'norm')


        prompt = []
        for b in range(x_enc.shape[0]):
            prompt_ = ("Analysing the following time series:")
            prompt.append(prompt_)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

       #print('input:', x_enc.shape)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32)) # P*dm
        
      #  print('after patch:', enc_out.shape)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # 1000(token_num）* 768
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) 
        enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
      #  print('after reprogram:', enc_out.shape)


        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state # B, P, 768
      #  print('after LLM:', dec_out.shape)
        last_hidden_states = dec_out

        dec_out = self.out_layer(dec_out) # B, P, 1
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out#, soh, recon
    
# x = batch, 50, 1
class RGHead(nn.Module):
    def __init__(self):
        super(RGHead, self).__init__()
        self.rghead = nn.Linear(52, 1)

    def forward(self, x):
        x = self.rghead(x.permute(0, 2, 1))
        return np.squeeze(x)
    
    
class SSHead(nn.Module):
    def __init__(self):
        super(SSHead, self).__init__()
        self.sshead = nn.Linear(52, 49)

    def forward(self, x):
        x = self.sshead(x.permute(0, 2, 1))
        return np.squeeze(x[:, :, :])



class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):

        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding
