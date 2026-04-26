import torch
import torch.nn.functional as F
from torch import nn


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video_features, text_features, labels):
        video_features = F.normalize(video_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        logits_v = (video_features @ text_features.T) / self.temperature
        logits_t = (text_features @ video_features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(video_features.device)

        mask_sum = mask.sum(dim=1, keepdim=True)
        target_probs = mask / torch.clamp(mask_sum, min=1e-8)

        log_probs_v = F.log_softmax(logits_v, dim=1)
        log_probs_t = F.log_softmax(logits_t, dim=1)

        loss_v = -torch.sum(target_probs * log_probs_v, dim=1).mean()
        loss_t = -torch.sum(target_probs * log_probs_t, dim=1).mean()

        return (loss_v + loss_t) / 2.0

def calculate_metrics(logits_v, logits_t):
    batch_size = logits_v.size(0)
    targets = torch.arange(batch_size).to(logits_v.device)

    def get_recall(logits, k):
        actual_k = min(k, logits.size(1))
        _, top_k_indices = logits.topk(actual_k, dim=1)
        correct = (top_k_indices == targets.view(-1, 1)).sum(dim=1)
        return correct.float().mean().item()

    def get_mean_rank(logits):
        sorted_indices = logits.argsort(dim=1, descending=True)
        ranks = (sorted_indices == targets.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
        return ranks.float().mean().item()

    return {
        'V2T_R1': get_recall(logits_v, 1),
        'V2T_R5': get_recall(logits_v, 5),
        'V2T_R10': get_recall(logits_v, 10),
        'V2T_Rank': get_mean_rank(logits_v),
        'T2V_R1': get_recall(logits_t, 1),
        'T2V_R5': get_recall(logits_t, 5),
        'T2V_R10': get_recall(logits_t, 10),
        'T2V_Rank': get_mean_rank(logits_t)
    }