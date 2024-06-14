import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from lib.models.feature_extractors import get_resnet18, get_moco_encoder
from lib.utils.preprocessing import preprocess_images_sequences


class To3Channels(nn.Module):

    def __init__(self,):
        super().__init__()
    def forward(self, img):
        return torch.cat([img, img, img], dim=0)

def main():
    # Base transforms for MoCo encoders
    transforms = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        #To3Channels(),
    ])

    def transform_func(img):
        # APPLIED TO ALL IMAGES IN THE SEQUENCE AT ONCE
        img_range = [150, 350]
        img = (img - img_range[0])/(img_range[1]-img_range[0])
        img = img.astype(np.float32)
        # img = torch.from_numpy(img).to("cuda").unsqueeze(0)
        img = transforms(img)
        return img

    model = get_moco_encoder("resnet34", "weights/moco_sequence/moco_r34_w6_projector/checkpoint_10000.pth")
    print(f"Encoder ready, model with {sum(p.numel() for p in model.parameters()):,} parameters")

    dataset_path =  "/fs9/datasets/typhoon-202404/wnp"

    preprocess_images_sequences(model,
                                "r34p_10k_w6",
                                transform_func,
                                device="cuda:1",
                                dataset_path=dataset_path)

if __name__ == "__main__":
    main()
