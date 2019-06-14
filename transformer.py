from enum import IntEnum
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


#  Enum to refer to dimensions
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


# Scaled dot-product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1) # get the size of the key
        assert q.size(-1) == d_k

        # compute the dot product between queries and keys for
        # each batch and position in the sequence
        attn = torch.bmm(q, k.transpose(Dim.seq, Dim.feature)) # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch

        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attn = attn / math.sqrt(d_k)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attn = torch.exp(attn)
        
        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) # (Batch, Seq, Feature)
        return output


# ## Multi-Head Attention
class AttentionHead(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super(AttentionHead, self).__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V, mask=mask)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is 
        # easier to understand this way
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model) 
    
    def forward(self, queries, keys, values, mask=None):
        x = [attn(queries, keys, values, mask=mask) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]
        
        # reconcatenate
        x = torch.cat(x, dim=Dim.feature) # (Batch, Seq, D_Feature * n_heads)
        x = self.projection(x) # (Batch, Seq, D_Model)
        return x


# ## Encoder
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        att = self.attn_head(x, x, x, mask=mask)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks=6, d_model=512,
                 n_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return x


# ## Decoder
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, 
                src_mask=None, tgt_mask=None):
        # Apply attention to inputs
        att = self.masked_attn_head(x, x, x, mask=src_mask)
        x = x + self.dropout(self.layer_norm1(att))
        # Apply attention to the encoder outputs and outputs of the previous layer
        att = self.attn_head(queries=x, keys=enc_out, values=enc_out, mask=tgt_mask)
        x = x + self.dropout(self.layer_norm2(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm3(pos))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks=6, d_model=512, 
                 d_ff=2048, n_heads=8, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.position_embedding = PositionalEmbedding(d_model)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
        
    def forward(self, x, enc_out, 
                src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


# ## Positional Encoding and Embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)
         
    def forward(self, x):
        return self.weight[:, :x.size(1), :] # (1, Seq, Feature)

class Embeddings(nn.Module):
    """Putting together embedding layer and PositionalEmbedding layer"""
    def __init__(self, d_model, vocab, max_len=512):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.positional_embed = PositionalEmbedding(d_model, max_len)
        self.d_model = d_model
    
    def forward(self, x):
        return self.positional_embed(self.embed(x) * np.sqrt(self.d_model))


# ## Generator
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ## Transformer model put together
class Transformer(nn.Module):
    """The transformer architecture which combines all the components"""
    def __init__(self, encoder, decoder, src_embedding, tgt_embedding, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_op = self.encoder(self.src_embedding(src), src_mask)
        return self.decoder(self.tgt_embedding(tgt), encoder_op, 
                            src_mask=src_mask, tgt_mask=tgt_mask)


# Function to build the model and initialize parameters
def buildTransformer(src_vocab, tgt_vocab, n_blocks=6, 
               d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
    model = Transformer(
        TransformerEncoder(n_blocks, d_model, n_heads, d_ff, dropout),
        TransformerDecoder(n_blocks, d_model, n_heads, d_ff, dropout),
        PositionalEmbedding(d_model),
        Embeddings(d_model, tgt_vocab),
        Generator()
    )
    
    # Initialize parameters with Glorot transform
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform(param)
    return model

