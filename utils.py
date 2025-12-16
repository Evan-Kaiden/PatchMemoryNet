import torch
import torch.nn as nn
import torch.nn.functional as F

# https://arxiv.org/pdf/2004.11362v1 (Supervised Contrastive Learning)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats, labels):
        device = feats.device
        N = feats.size(0)

        feats = F.normalize(feats, dim=1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(feats @ feats.T, self.temperature)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(N, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_counts = mask.sum(dim=1)
        valid = pos_counts > 0
        mean_log_prob_pos = torch.zeros_like(pos_counts, device=device)
        mean_log_prob_pos[valid] = (mask[valid] * log_prob[valid]).sum(dim=1) / pos_counts[valid]

        loss = -mean_log_prob_pos[valid].mean()

        return loss
    
def gumbel_topk_st(logits: torch.Tensor, k: int, tau: float):
    """
    logits: (B, T)
    returns:
      weights: (B, T)  # k-hot forward, soft-grad backward
      idx:     (B, k)
    """
    # sample k indices without replacement via Gumbel perturbation
    g = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + g
    idx = y.topk(k, dim=-1).indices  # (B, k)

    w_hard = torch.zeros_like(logits).scatter_(1, idx, 1.0)      # k-hot
    w_soft = F.softmax(logits / tau, dim=-1)                     # smooth probs
    weights = w_hard + (w_soft - w_soft.detach())                # straight-through
    return weights, idx
