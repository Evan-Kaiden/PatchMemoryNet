import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sim_classifier import pred_by_score
from dataloader import testloader, memloader

from models.matchnet import Matcher
from samplier import extract_patches


def undoNorm(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

    if img.dim() == 3:
        return img * std + mean
    elif img.dim() == 4:
        return img * std[None] + mean[None]

def top_k_contributors(image : torch.Tensor, model : Matcher, memory : torch.Tensor, memory_cls : torch.Tensor, k=10):
    assert len(image.shape) == 4
    B = image.size(0)
    assert B == 1

    img_patches = model.extractor(image)
    mem_patches = model.extractor(memory)

    _, Tq, C, H, W = img_patches.shape
    M, Tm, _, _, _ = mem_patches.shape

    img_patches = img_patches.reshape(Tq, C, H, W).contiguous()
    mem_patches = mem_patches.reshape(M * Tm, C, H, W).contiguous()

    img_embeds = model.encoder(img_patches)
    mem_embeds = model.encoder(mem_patches)

    sim = img_embeds @ mem_embeds.t() / model.temperature

    with torch.no_grad():
        logits = model.predict(image, memory, memory_cls)
    pred_cls = logits.argmax(dim=1).item()

    mem_labels = memory_cls.repeat_interleave(Tm)
    mask = (mem_labels == pred_cls)

    masked_sim = sim[:, mask]
    mem_idx_all = mask.nonzero(as_tuple=False).view(-1)

    _, L = masked_sim.shape

    sim_flat = masked_sim.reshape(-1) 

    k_eff = min(k, sim_flat.numel())
    scores, top_indices = torch.topk(sim_flat, k_eff, largest=True)

    img_patch_idx = top_indices // L
    mem_idx_in_mask = top_indices % L

    mem_patch_idx = mem_idx_all[mem_idx_in_mask]

    img_patch_idx, mem_patch_idx, scores = list(img_patch_idx), list(mem_patch_idx), list(scores)


    relations = []

    for img_idx, mem_idx, score in zip(img_patch_idx, mem_patch_idx, scores):
        img_patch = img_patches[img_idx]
        mem_patch = mem_patches[mem_idx]
        
        relations.append((img_patch, mem_patch, score))

    return relations


def extractor(img):
    return extract_patches(img, kernel_size=12, stride=3)

model = Matcher(10,extractor)

image = next(iter(testloader))[0][0:1, : ,: ,:]
memory, cls = next(iter(memloader))

# relations = top_k_contributors(image, model, memory, cls)


# img, mem, score = relations[0]

# img = undoNorm(img)
# mem = undoNorm(mem)

# img = img.permute(1, 2, 0).cpu().numpy()
# mem = mem.permute(1, 2, 0).cpu().numpy()

# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# fig.suptitle(f"Sim Score {score}")

# axes[0].imshow(img)
# axes[0].set_title("Input patch")
# axes[0].axis("off")

# axes[1].imshow(mem)
# axes[1].set_title("Memory patch")
# axes[1].axis("off")

# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import math
import torch

import torch

def top_k_contributors(
    image: torch.Tensor,
    model: Matcher,
    memory: torch.Tensor,
    memory_cls: torch.Tensor,
    k: int = 10
):
    assert len(image.shape) == 4
    B = image.size(0)
    assert B == 1

    img_patches = model.extractor(image)      # (1, Tq, C, H, W)
    mem_patches = model.extractor(memory)     # (M, Tm, C, H, W)

    _, Tq, C, H, W = img_patches.shape
    M, Tm, _, _, _ = mem_patches.shape

    img_patches_flat = img_patches.reshape(Tq, C, H, W).contiguous()
    mem_patches_flat = mem_patches.reshape(M * Tm, C, H, W).contiguous()

    img_embeds = model.encoder(img_patches_flat)   # (Tq, D)
    mem_embeds = model.encoder(mem_patches_flat)   # (M*Tm, D)

    sim = img_embeds @ mem_embeds.t() / model.temperature

    with torch.no_grad():
        logits = model.predict(image, memory, memory_cls)
    pred_cls = logits.argmax(dim=1).item()

    mem_labels = memory_cls.view(M, 1).expand(M, Tm).reshape(-1)  # (M*Tm,)
    mask = (mem_labels == pred_cls)

    if not mask.any():
        masked_sim = sim
        mem_idx_all = torch.arange(M * Tm, device=sim.device)
    else:
        masked_sim = sim[:, mask]                     # (Tq, L)
        mem_idx_all = mask.nonzero(as_tuple=False).view(-1)  # (L,)

    Tq_eff, L = masked_sim.shape
    sim_flat = masked_sim.reshape(-1)

    k_eff = min(k, sim_flat.numel())
    scores, top_indices = torch.topk(sim_flat, k_eff, largest=True)

    img_patch_idx = (top_indices // L)
    mem_idx_in_mask = (top_indices % L)
    mem_patch_idx = mem_idx_all[mem_idx_in_mask]

    relations = []
    for img_idx, mem_idx, score in zip(img_patch_idx.tolist(),
                                       mem_patch_idx.tolist(),
                                       scores.tolist()):
        relations.append({
            "score": score,
            "img_patch_index": img_idx,        # 0..Tq-1
            "mem_patch_global_index": mem_idx, # 0..(M*Tm-1)
            "img_patch": img_patches_flat[img_idx],      # (C,H,W)
            "mem_patch": mem_patches_flat[mem_idx],      # (C,H,W)
        })

    return relations


def show_relation_on_images(
    image: torch.Tensor,
    memory: torch.Tensor,
    relation: dict,
    kernel_size: int,
    stride: int,
    denorm_fn=None,
    title: str = None,
):
    """
    Visualize one relation:
      - left: input image with the patch highlighted
      - right: matching memory image with its patch highlighted

    image: (1, 3, H, W)
    memory: (M, 3, H, W)
    relation: one dict from top_k_contributors
    kernel_size, stride: same values used in extract_patches
    """

    img_idx = relation["img_patch_index"]
    mem_global_idx = relation["mem_patch_global_index"]

    # shapes
    _, _, H_img, W_img = image.shape
    M, _, H_mem, W_mem = memory.shape

    # how many patches per dim (input image)
    n_h_img = (H_img - kernel_size) // stride + 1
    n_w_img = (W_img - kernel_size) // stride + 1

    img_row = img_idx // n_w_img
    img_col = img_idx % n_w_img

    y_img = img_row * stride
    x_img = img_col * stride

    # memory index -> which image + which patch inside that image
    # for each memory image, there are Tm patches, but we don't need Tm explicitly
    # we can recompute n_h_mem, n_w_mem the same way
    n_h_mem = (H_mem - kernel_size) // stride + 1
    n_w_mem = (W_mem - kernel_size) // stride + 1
    Tm = n_h_mem * n_w_mem

    mem_image_index = mem_global_idx // Tm
    mem_patch_index = mem_global_idx % Tm

    mem_row = mem_patch_index // n_w_mem
    mem_col = mem_patch_index % n_w_mem

    y_mem = mem_row * stride
    x_mem = mem_col * stride

    # prepare images
    img_vis = image[0]      # (3,H,W)
    mem_vis = memory[mem_image_index]  # (3,H,W)

    if denorm_fn is not None:
        img_vis = denorm_fn(img_vis)
        mem_vis = denorm_fn(mem_vis)

    img_vis = img_vis.permute(1, 2, 0).detach().cpu().numpy()
    mem_vis = mem_vis.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # input image + box
    axes[0].imshow(img_vis)
    axes[0].add_patch(Rectangle(
        (x_img, y_img),
        kernel_size,
        kernel_size,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    ))
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # memory image + box
    axes[1].imshow(mem_vis)
    axes[1].add_patch(Rectangle(
        (x_mem, y_mem),
        kernel_size,
        kernel_size,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    ))
    axes[1].set_title(f"Memory image {mem_image_index}")
    axes[1].axis("off")

    if title is None:
        title = f"Match score: {relation['score']:.3f}"

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



relations = top_k_contributors(image, model, memory, cls, k=10)

# e.g. visualize the best match
best_relation = relations[0]

show_relation_on_images(
    image,
    memory,
    best_relation,
    kernel_size=12,
    stride=3,
    denorm_fn=undoNorm,  # or None
)



