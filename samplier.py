import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def extract_patch(img: torch.tensor, x : torch.tensor, y : torch.tensor, dx : torch.tensor, dy : torch.tensor):
    B, C, H, W = img.shape
    x = x.squeeze()
    y = y.squeeze()

    yy, xx = torch.meshgrid(
        torch.arange(-(dy//2), dy - dy//2, device=img.device),
        torch.arange(-(dx//2), dx - dx//2, device=img.device),
        indexing='ij'
    )  
   
    xs = (x[:, None, None] + xx).clamp(0, W-1)
    ys = (y[:, None, None] + yy).clamp(0, H-1)

    b = torch.arange(B, device=img.device)[:, None, None]
    patch = img[b, :, ys, xs] 

    return patch


def sample_patches(img: torch.Tensor, prob_dist: torch.Tensor, n_patches: int, patch_size):
    """Samples N patches of size (patch_size, patch_size) acoording to a probabliy distrubtion"""
    assert torch.min(prob_dist) >= 0 and prob_dist.sum(dim=1, keepdim=True).all() == 1
    assert img.shape[2] - prob_dist.shape[1] >= patch_size and img.shape[3] - prob_dist.shape[2] >= patch_size
    
    _, H, W = prob_dist.shape
    prob_dist = prob_dist.flatten(start_dim=1)
    dist = Categorical(prob_dist)

    built = []

    for _ in range(n_patches):
        sample = dist.sample()
        row, col = sample // H, sample % W

        patches = extract_patch(img, row, col, patch_size, patch_size)
        built.append(patches)

    built = torch.stack(built, dim=1)
    return built

def extract_patches(x : torch.Tensor, kernel_size : int, stride : int = 1):
    """Return all patches of an image given a patch size and a stride"""
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).reshape(-1, C, kernel_size, kernel_size)
    T = patches.size(0) // B

    patches = patches.reshape(B, T, C, kernel_size, kernel_size)

    return patches.contiguous()
