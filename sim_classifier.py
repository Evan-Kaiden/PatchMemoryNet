import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def pred_by_vote(sim, cls, B):
    best_idx = sim.argmax(dim=1)             
    patch_preds = cls[best_idx]

    patch_preds = patch_preds.view(B, -1) 
    preds, _ = torch.mode(patch_preds.to('cpu'), dim=1)

    return preds

def pred_by_score(sim, cls, num_classes, B, N):
    cls_one_hot = F.one_hot(cls, num_classes=num_classes).float()
    sim_per_class = sim @ cls_one_hot
    sim_per_class = sim_per_class.view(B, N, num_classes)
    class_scores = sim_per_class.sum(dim=1)
    probs = F.softmax(class_scores, dim=1)  
    preds = probs.argmax(dim=1)   

    return preds, class_scores

def sim_classify(encoder : nn.Module, testloader: DataLoader, memloader : DataLoader, extractor, device=None):
    encoder.eval()
    
    correct = 0
    total = 0

    mem_iter = iter(memloader)

    for image, target in testloader:
        B = image.size(0)
        image, target = image.to(device), target.to(device)

        try:
            mem_images, cls = next(mem_iter)
            mem_images, cls = mem_images.to(device), cls.to(device)
        except StopIteration:
            mem_iter = iter(memloader)
            mem_images, cls = next(mem_iter)
            mem_images, cls = mem_images.to(device), cls.to(device)


        image_patches = extractor(image)
        mem_patches = extractor(mem_images)

        N = image_patches.size(1)
        cls = torch.repeat_interleave(cls, N)

        image_patches = torch.flatten(image_patches, start_dim=0, end_dim=1)
        mem_patches = torch.flatten(mem_patches, start_dim=0, end_dim=1)

        image_feats = encoder(image_patches)
        mem_feats = encoder(mem_patches)

        image_feats = image_feats.view(image_feats.size(0), -1) 
        mem_feats = mem_feats.view(mem_feats.size(0), -1)

        image_feats_norm = image_feats / image_feats.norm(dim=1, keepdim=True)
        mem_feats_norm = mem_feats / mem_feats.norm(dim=1, keepdim=True)

        sim = torch.matmul(image_feats_norm, mem_feats_norm.t())

        preds, _ = pred_by_score(sim, cls, num_classes=10, B=B, N=N)
        # preds = pred_by_vote(sim, cls, B)
        correct += (preds == target.to(preds.device)).sum().item()
        total += B
        
    print(f"ACC {correct / total} ({correct}/{total})")



        