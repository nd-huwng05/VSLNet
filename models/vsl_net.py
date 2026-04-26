import numpy as np
import torch
from torch import nn
from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder

class VSLContrastiveNet(nn.Module):
    def __init__(self, vocab_size=400, embedding_size=256, initial_temperature=0.07):
        super(VSLContrastiveNet, self).__init__()
        self.video_encoder = VideoEncoder(embedding_size=embedding_size)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embedding_size=embedding_size)
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/initial_temperature))

    def forward(self, x, labels_idx):
        video_embedding = self.video_encoder(x)
        text_embedding = self.text_encoder(labels_idx)
        logit_scale = self.logit_scale.exp()
        logit_per_video = logit_scale * video_embedding@text_embedding.T
        logit_per_text = logit_per_video.T

        return logit_per_video, logit_per_text
