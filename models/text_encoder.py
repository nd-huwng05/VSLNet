from torch import nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=400, d_model=256, embedding_size=256):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_size),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.proj(x)
        embedding = F.normalize(x, dim=1, p=2)
        return embedding