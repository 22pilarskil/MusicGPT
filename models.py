import torch.nn as nn
import torch

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(RelativePositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(2*max_len-1, d_model)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position = position.unsqueeze(0).expand_as(x)  # [bs, seq_len]
        dists = position.unsqueeze(-1) - position.unsqueeze(-2)
        dists = dists + self.max_len - 1  # Shift values to be positive
        return self.embedding(dists)

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_length):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = RelativePositionalEmbedding(embed_dim, max_length)

        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length

    def forward(self, x):
        x_embed = self.embedding(x)
        pos_embed = self.pos_embedding(x)
        x = x_embed + pos_embed

        mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).bool().to(x.device)
        x = self.transformer(x, x, tgt_mask=mask)
        return self.fc(x)
