import torch
import torch.nn.functional as F

def bilinear_sampler(img, coords):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


def coords_grid(batch: int, height: int, width: int):
    coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
