import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def movi_label_reading(image_path, max_num_obj, keys_to_log=None):
    with open(image_path, 'rb') as f:
        label_dict = json.load(f)

    num_obj = len(label_dict['visibility'])
    labels = {'num_obj': num_obj}
    for k, v in label_dict.items():
        if np.array(v).ndim >= 3:
            # flatten the last two dimensions
            v = np.array(v)
            v = v.reshape(v.shape[0], -1).tolist()
        if k == 'visibility':
            continue
        if keys_to_log is None:
            labels[k] = torch.tensor(
                [i for i in v] + [v[-1] for _ in range(max_num_obj - num_obj)]
            )
        else:
            if k in keys_to_log:
                labels[k] = torch.tensor(
                    [i for i in v] + [v[-1] for _ in range(max_num_obj - num_obj)]
                )
    labels['visibility'] = torch.tensor(
        [i for i in label_dict['visibility']] +
        [0 for _ in range(max_num_obj - num_obj)]
    )

    return labels


class MOViDataset(Dataset):

    def __init__(
        self,
        root: str,
        img_size: int = 256,
        split: str = "train",
        random_flip: bool = False,
        read_labels: bool = False,
        max_num_obj: int = 23,
        keys_to_log: List[str] = None,
    ):
        """Dataset from a folder of images.

        Args:
            root (str): Path to images.
            img_size (int, optional): Image size. Defaults to 256.
            split (str, optional): "train", "validation", or "test".
            rng_seed (int, optional): Random seed. Defaults to 1234.
        """

        self.img_size = (img_size, img_size)

        self.trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        root = Path(root) / split
        images = list(root.glob(f"**/*.jpg"))

        self.images = images
        self.random_flip = random_flip

        if read_labels:
            assert not random_flip, "cannot random flip"

        self.read_labels = read_labels
        self.max_num_obj = max_num_obj
        self.keys_to_log = keys_to_log

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]

        msk = Image.open(file.parent / (file.stem + "_mask.png"))
        img = Image.open(file).convert('RGB')

        img = img.resize(self.img_size, resample=Image.BILINEAR)
        msk = msk.resize(self.img_size, resample=Image.NEAREST)

        if self.random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.trans_img(img)
        msk = torch.from_numpy(np.array(msk)).long()

        output = {"image": img, "mask": msk}

        if self.read_labels:
            metafile = file.parent / (file.stem + "_instances.json")
            output["labels"] = movi_label_reading(metafile, self.max_num_obj, self.keys_to_log)

        return output
