import torch
import torch.nn as nn

from matchnet import Matcher
from samplier import extract_patches
from dataloader import testloader, memloader

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def undoNorm(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

    if img.dim() == 3:
        return img * std + mean
    elif img.dim() == 4:
        return img * std[None] + mean[None]

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

    img_patches = model.extractor(image)
    mem_patches = model.extractor(memory)

    _, Tq, C, H, W = img_patches.shape
    M, Tm, _, _, _ = mem_patches.shape

    img_patches_flat = img_patches.reshape(Tq, C, H, W).contiguous()
    mem_patches_flat = mem_patches.reshape(M * Tm, C, H, W).contiguous()

    img_embeds = model.encoder(img_patches_flat)
    mem_embeds = model.encoder(mem_patches_flat)

    sim = img_embeds @ mem_embeds.t() / model.temperature

    with torch.no_grad():
        logits = model.predict(image, memory, memory_cls)
    pred_cls = logits.argmax(dim=1).item()

    mem_labels = memory_cls.view(M, 1).expand(M, Tm).reshape(-1)
    mask = (mem_labels == pred_cls)

    if not mask.any():
        masked_sim = sim
        mem_idx_all = torch.arange(M * Tm, device=sim.device)
    else:
        masked_sim = sim[:, mask]
        mem_idx_all = mask.nonzero(as_tuple=False).view(-1)

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
            "img_patch_index": img_idx,
            "mem_patch_global_index": mem_idx,
            "img_patch": img_patches_flat[img_idx],
            "mem_patch": mem_patches_flat[mem_idx],
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


    img_idx = relation["img_patch_index"]
    mem_global_idx = relation["mem_patch_global_index"]

    _, _, H_img, W_img = image.shape
    M, _, H_mem, W_mem = memory.shape

    n_h_img = (H_img - kernel_size) // stride + 1
    n_w_img = (W_img - kernel_size) // stride + 1

    img_row = img_idx // n_w_img
    img_col = img_idx % n_w_img

    y_img = img_row * stride
    x_img = img_col * stride

    n_h_mem = (H_mem - kernel_size) // stride + 1
    n_w_mem = (W_mem - kernel_size) // stride + 1
    Tm = n_h_mem * n_w_mem

    mem_image_index = mem_global_idx // Tm
    mem_patch_index = mem_global_idx % Tm

    mem_row = mem_patch_index // n_w_mem
    mem_col = mem_patch_index % n_w_mem

    y_mem = mem_row * stride
    x_mem = mem_col * stride

    img_vis = image[0]
    mem_vis = memory[mem_image_index]

    if denorm_fn is not None:
        img_vis = denorm_fn(img_vis)
        mem_vis = denorm_fn(mem_vis)

    img_vis = img_vis.permute(1, 2, 0).detach().cpu().numpy()
    mem_vis = mem_vis.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

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



def extractor(img):
    return extract_patches(img, kernel_size=12, stride=6)

model = Matcher(10,extractor)

state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict=state_dict)

import random

memory, cls = next(iter(memloader))
loader = iter(testloader)
while True:
    n = random.randint(0, 30)

    image, label = next(loader)
    image = image[n:n+1, : ,: ,:]
    label = label[n:n+1]
    logits = model.predict(image, memory, cls)
    # if torch.argmax(logits) == label:
        # break
    break

relations = top_k_contributors(image, model, memory, cls, k=10)

for relation in relations:
    show_relation_on_images(
        image,
        memory,
        relation,
        kernel_size=12,
        stride=3,
        denorm_fn=undoNorm,
    )



