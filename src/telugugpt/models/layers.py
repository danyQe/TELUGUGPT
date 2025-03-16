import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        return self.embedding(input) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_seq_len, d_model)

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_embedding):
        input_embedding = input_embedding + (self.pe[:, :input_embedding.shape[1], :]).requires_grad_(False)
        return self.dropout(input_embedding)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by number of heads"

        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask):
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        
        attention_score = attention_score.softmax(dim=-1)
        attention_score = self.dropout(attention_score)
        attention_output = attention_score @ value

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            attention_output.shape[0], -1, self.num_heads * self.d_k
        )
        return self.W_o(attention_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, input):
        return self.layer_2(self.dropout(torch.relu(self.layer_1(input))))

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(512))
        self.beta = nn.Parameter(torch.zeros(512))

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        return self.gamma * (input - mean) / (std + self.eps) + self.beta

class AddAndNorm(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm()

    def forward(self, input, sub_layer):
        return input + self.dropout(sub_layer(self.layer_norm(input))) 