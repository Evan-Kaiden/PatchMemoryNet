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
    k: int = 10,
    cls_filter: str = "pred", 
    true_label: int | None = None,
):
    assert image.ndim == 4 and image.size(0) == 1

    img_patches = model.extractor(image)
    mem_patches = model.extractor(memory)

    _, Tq, C, H, W = img_patches.shape
    M, Tm, _, _, _ = mem_patches.shape

    img_patches_flat = img_patches.reshape(Tq, C, H, W).contiguous()
    mem_patches_flat = mem_patches.reshape(M * Tm, C, H, W).contiguous()

    img_embeds = model.encoder(img_patches_flat)
    mem_embeds = model.encoder(mem_patches_flat)

    sim = (img_embeds @ mem_embeds.t()) / model.temperature 

    device = sim.device

    if memory_cls.ndim == 2:
        memory_cls = memory_cls.argmax(dim=1)
    memory_cls = memory_cls.view(-1).long().to(device)
    assert memory_cls.numel() == M, f"memory_cls must have length M. got {memory_cls.shape}, M={M}"

    if cls_filter == "true":
        assert true_label is not None, "true_label must be provided when cls_filter='true'"
        target_cls = int(true_label)
    else:
        with torch.no_grad():
            logits = model.predict(image, memory, memory_cls)
        target_cls = int(logits.argmax(dim=1).item())

    mem_labels = memory_cls.repeat_interleave(Tm)
    assert mem_labels.shape[0] == M * Tm

    mask = (mem_labels == target_cls)

    if mask.any():
        masked_sim = sim[:, mask]
        mem_idx_all = mask.nonzero(as_tuple=False).view(-1)
    else:
        masked_sim = sim 
        mem_idx_all = torch.arange(M * Tm, device=device)

    sim_flat = masked_sim.reshape(-1)
    k_eff = min(k, sim_flat.numel())
    scores, top_indices = torch.topk(sim_flat, k_eff, largest=True)

    L = masked_sim.shape[1]
    img_patch_idx = (top_indices // L)
    mem_idx_in_mask = (top_indices % L)
    mem_patch_idx = mem_idx_all[mem_idx_in_mask]

    if mask.any():
        picked_image_idx = (mem_patch_idx // Tm)  # [k_eff]
        assert (memory_cls[picked_image_idx] == target_cls).all(), \
            "Selected memory patches are not all from the filtered class!"

    relations = []
    for img_idx, mem_idx, score in zip(img_patch_idx.tolist(),
                                       mem_patch_idx.tolist(),
                                       scores.tolist()):
        mem_image_idx = mem_idx // Tm
        mem_patch_in_image = mem_idx % Tm

        relations.append({
            "score": float(score),
            "target_cls": target_cls,
            "img_patch_index": int(img_idx),
            "mem_patch_global_index": int(mem_idx),
            "mem_image_index": int(mem_image_idx),
            "mem_patch_index_in_image": int(mem_patch_in_image),
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

    img_patch_idx = int(relation["img_patch_index"])
    mem_image_index = int(relation["mem_image_index"])
    mem_patch_idx_in_image = int(relation["mem_patch_index_in_image"])

    _, _, H_img, W_img = image.shape
    _, _, H_mem, W_mem = memory.shape

    n_h_img = (H_img - kernel_size) // stride + 1
    n_w_img = (W_img - kernel_size) // stride + 1

    n_h_mem = (H_mem - kernel_size) // stride + 1
    n_w_mem = (W_mem - kernel_size) // stride + 1

    
    Tq_expected = n_h_img * n_w_img
    Tm_expected = n_h_mem * n_w_mem
    assert img_patch_idx < Tq_expected, f"img_patch_idx {img_patch_idx} out of range {Tq_expected}"
    assert mem_patch_idx_in_image < Tm_expected, f"mem_patch_idx {mem_patch_idx_in_image} out of range {Tm_expected}"

    img_row = img_patch_idx // n_w_img
    img_col = img_patch_idx % n_w_img
    y_img = img_row * stride
    x_img = img_col * stride

    mem_row = mem_patch_idx_in_image // n_w_mem
    mem_col = mem_patch_idx_in_image % n_w_mem
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
    axes[0].add_patch(Rectangle((x_img, y_img), kernel_size, kernel_size,
                                linewidth=2, edgecolor="r", facecolor="none"))
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(mem_vis)
    axes[1].add_patch(Rectangle((x_mem, y_mem), kernel_size, kernel_size,
                                linewidth=2, edgecolor="r", facecolor="none"))
    axes[1].set_title(f"Memory image {mem_image_index}")
    axes[1].axis("off")

    if title is None:
        title = f"Match score: {relation['score']:.3f}"

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



kernel = 18
stride = 3

def extractor(img):
    return extract_patches(img, kernel_size=kernel, stride=stride)

model = Matcher(10, extractor).to("mps")
state_dict = torch.load("model.pth", map_location="mps")
model.load_state_dict(state_dict)

memory, cls = next(iter(memloader))

loader = iter(testloader)
print("finding correctly classified image")
while True:
    image, label = next(loader)
    with torch.no_grad():
        logits = model.predict(image.to("mps"), memory.to("mps"), cls.to("mps"))
    mask = (torch.argmax(logits, dim=-1).cpu() == label)
    if mask.any():
        image = image[mask][0:1]
        break

image = image.cpu()
model = model.to("cpu")
model.eval()

print("finding top contributors")
with torch.no_grad():
    relations = top_k_contributors(image, model, memory, cls, k=10)

print("displaying relations")
for relation in relations:
    show_relation_on_images(
        image,
        memory,
        relation,
        kernel_size=kernel,
        stride=stride,
        denorm_fn=undoNorm,
    )

