import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.3, max_len=500):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1) # matrix(T) 1,256

        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)) # PE 1,256
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term) # sin, position * div_term = 256 * 256
        pe[0, :, 1::2] = torch.cos(position * div_term) # cos

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VideoEncoder(nn.Module):
    def __init__(self, num_joints=42, in_chans=3, d_spatial=64, d_temporal=256, num_frames=64, dropout=0.1, embedding_size=256):
        super(VideoEncoder, self).__init__()
        self.spatial_proj = nn.Linear(in_chans, d_spatial)
        self.spatial_cls_token = nn.Parameter(torch.randn(1,1,d_spatial))
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_joints + 1, d_spatial))

        spatial_layer = nn.TransformerEncoderLayer(d_model=d_spatial, nhead=4, dim_feedforward=256, batch_first=True, dropout=dropout)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, 2)

        self.temporal_proj = nn.Linear(d_spatial, d_temporal)
        self.temporal_cls_token = nn.Parameter(torch.randn(1,1,d_temporal))
        self.temporal_pos_embed = TemporalPositionalEncoding(d_temporal, dropout=dropout, max_len=num_frames + 1)

        temporal_layer = nn.TransformerEncoderLayer(d_model=d_temporal, nhead=8, dim_feedforward=1024, batch_first=True, dropout=dropout)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, 2)

        self.head = nn.Sequential(
            nn.LayerNorm(d_temporal),
            nn.Linear(d_temporal, embedding_size),
        )

    def forward(self, x):
        B, T, _ = x.shape  # (B, 64, 126), T is num frames

        x = x.view(B * T, 42, 3)

        x = self.spatial_proj(x) # 3 dim -> 64 dim, (B*T, 42, 64)
        cls_spatial = self.spatial_cls_token.expand(B * T, -1, -1) #(B*T, 1, 64)
        x = torch.cat((cls_spatial, x), dim=1) #(B*T, 43, 64)
        x = x + self.spatial_pos_embed # Parameter to learn self.spatial_pos_embed has (1, 43, 64)
        x = self.spatial_transformer(x)
        x_spatial = x[:, 0, :] # Get cls_spatial (B*T, 64)

        x_temporal = x_spatial.view(B, T, -1) #(B,T,64)

        x_temporal = self.temporal_proj(x_temporal) #(B,T,256)
        cls_temporal = self.temporal_cls_token.expand(B, -1, -1) #(B, 1, 256)
        x_temporal = torch.cat((cls_temporal, x_temporal), dim=1) #(B, T+1, 256)
        x_temporal = self.temporal_pos_embed(x_temporal)
        x_temporal = self.temporal_transformer(x_temporal)
        final_cls = x_temporal[:, 0, :]

        embedding = self.head(final_cls)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

if __name__ == 'main':
    model = VideoEncoder(num_joints=42, in_chans=1)
    model.train()