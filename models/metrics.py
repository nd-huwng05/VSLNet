import torch
import torch.nn.functional as F

def contrastive_loss(logits_v, logits_t):
    batch_size = logits_v.size(0)
    targets = torch.arange(batch_size).to(logits_v.device)

    loss_video = F.cross_entropy(logits_v, targets)
    loss_text = F.cross_entropy(logits_t, targets)
    return (loss_video + loss_text) / 2.0

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