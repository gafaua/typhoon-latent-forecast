from os import makedirs

import numpy as np
import torch
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SimpleSeqDataset(Dataset):
    def __init__(self, prefix) -> None:
        # prefix is the path to the Digital Typhoon Dataset

        super().__init__()
        self.dataset = DigitalTyphoonDataset(
                get_images_by_sequence=True,
                labels=[],
                filter_func= None,
                ignore_list=[],
                transform=None,
                verbose=False,
                image_dir=f"{prefix}/image/",
                metadata_dir=f"{prefix}/metadata/",
                metadata_json=f"{prefix}/metadata.json",
            )

    def __getitem__(self, index):
        return self.dataset.get_ith_sequence(index)

    def __len__(self):
        return len(self.dataset)

def preprocess_images_sequences(model, out_dir, transform, device, dataset_path):
    """
    Projects all sequences of the Digital Typhoon Dataset

    Args:
        model (nn.Module): Image encoder
        out_dir (str): Directory to write all feature files
        transform (function): transform function to be applied to sequences
        device (str): device to use to for processung
    """
    makedirs(out_dir, exist_ok=True)
    print(f"Writing feature files to {out_dir}")
    dataset = SimpleSeqDataset(dataset_path)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=16,
                        shuffle=False,
                        collate_fn=lambda x: x)

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for seq in tqdm(loader):
            seq = seq[0]
            images = seq.get_all_images_in_sequence()
            names = np.array([str(image.image_filepath).split("/")[-1].split(".")[0] for image in images])

            images = torch.stack([transform(image.image()) for image in images])

            images = images.to(device)

            features = model(images).cpu().numpy()
            np.savez(f"{out_dir}/{seq.sequence_str}", features, names)

