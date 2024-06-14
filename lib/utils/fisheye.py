import torch
import torch.nn.functional as F


def fisheye_grid(width, height, alpha):
    # Create a grid of normalized coordinates
    x, y = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height), indexing="ij")
    coords = torch.stack((y, x), dim=-1)

    # Apply fisheye transformation to the coordinates
    r = torch.sqrt(coords[:, :, 0]**2 + coords[:, :, 1]**2)
    radial_scale = torch.pow(r, alpha)#(1 - torch.pow(r, alpha)) / r
    radial_scale[r == 0] = 1.0
    fisheye_coords = coords * torch.unsqueeze(radial_scale, -1)

    # Clamp the transformed coordinates to [-1, 1] range
    fisheye_coords = torch.clamp(fisheye_coords, min=-1, max=1)

    return fisheye_coords

class FishEye(torch.nn.Module):
    def __init__(self, size, alpha):
        super().__init__()
        self.grid = fisheye_grid(size, size, alpha)

    def forward(self, img):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        B, _, _, _ = img.shape
        fish = F.grid_sample(img, self.grid.unsqueeze(0).repeat(B, 1, 1, 1), align_corners=True).squeeze(0)
        return fish

