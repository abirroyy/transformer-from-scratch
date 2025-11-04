import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
pad_idx = 0

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model) #Shape [seq_len, d_model] --> [batch, seq_len, d_model]
        position = torch.arange(0,max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (- math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0)) # Shpae: [batch, seq_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model, bias= False)
        self.wk = nn.Linear(d_model, d_model, bias= False)
        self.wv = nn.Linear(d_model, d_model, bias= False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask = None, padding_mask = None):
        B, L, _ = Q.size()

        Q = self.wq(Q).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # [B, self.num_heads, L, self.head_dim]
        K = self.wk(K).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # d_model = 8 ---->  head1 = 4 ---  head2 = 4
        V = self.wv(V).view(B, L, self.num_heads, self.head_dim).transpose(1,2)

        attn_scores = (Q @ K.transpose(-2,-1) /(math.sqrt(self.head_dim)))

        if attn_mask is not None:
            attn_scores+= attn_mask

        if padding_mask is not None:
            mask = padding_mask[:, None, None, : ].bool() # 5--- The winter is--- 3 + 2 = 5 The winter is --> False, 2 ---> True 
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores,dim= -1)
        attn_weights = (attn_weights @ V)

        output = attn_weights.transpose(1,2).contiguous().view(B, L, self.d_model)
        return output
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads,d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.feed = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_padding_mask = None):
        x = self.norm1(x + self.mha(x, x, x, padding_mask = src_padding_mask))
        x = self.norm2(x + self.feed(x))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask = None, tgt_padding_mask = None, src_padding_mask = None):
        x = self.norm1(x + self.self_attn(x,x,x, attn_mask = tgt_mask, padding_mask = tgt_padding_mask))
        x = self.norm2(x + self.cross_attn(memory, memory, x, padding_mask = src_padding_mask))
        x = self.norm3( x + self.feed(x))
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, num_heads, num_layers, d_model, d_ff, dropout, max_seq_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layer = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layer = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size, bias= False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size))
        mask = mask.float().masked_fill(mask==0, float("-inf")).masked_fill(mask==1, float(0.0)).transpose(0,1)
        return mask
    
    def forward(self, src, tgt):
        src_mask = None
        tgt_mask = self.subsequent_mask(tgt.size(1)).to(device)

        src_padding_mask = (src==pad_idx)
        tgt_padding_mask = (tgt==pad_idx)

        #encoder part

        x = self.dropout(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layer:
            x = layer(x, src_padding_mask= src_padding_mask)
        
        memory = x

        #decoder part

        y = self.dropout(self.embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layer:
            y = layer(x, memory, tgt_mask = tgt_mask, tgt_padding_mask = tgt_padding_mask, src_padding_mask = src_padding_mask)

        output = self.out_proj(y)
        output = F.softmax(output, dim=-1)

        return output
    

if __name__ == "__main__":
    vocab_size = 5000
    d_model = 768
    num_layers = 6
    num_heads = 64
    max_seq_len = 2000
    d_ff = 2048
    dropout = 0.05

    transformer = TransformerBlock(vocab_size, num_heads, num_layers, d_model, d_ff, dropout, max_seq_len).to(device)

    src = torch.randint(1,11, (5,5)).to(device)
    tgt = torch.randint(1,11, (5,5)).to(device)

    src[:, -1:] == pad_idx
    tgt[:, -1:] == pad_idx

    output = transformer(src, tgt)
    print(output)
    print(f"Input Shape: {src.shape}")
    print(f"Ouput Shape: {output.shape}")

    gc.collect()
    torch.cuda.empty_cache()


