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
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eye(labels.size(0))

        anchor_dot_contrast = torch.div(
            torch.matmul(feats, feats.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=-1, keepdim=True)
        logits = anchor_dot_contrast - logits_max

        mask = torch.tile(mask, (2, 2))

        logits_mask = torch.scatter(torch.ones_like(mask), 1, 
                                    torch.arange(feats.shape[0]).view(-1, 1).long().to(device), 
                                    0)

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss