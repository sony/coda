from pathlib import Path
from typing import Tuple

from numpy.random import RandomState
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class GlobDataset(Dataset):

    def __init__(
        self,
        root: str,
        metafile: str = None,
        img_size: int = 256,
        data_portion: Tuple = (0, 1),
        img_ext: str = ".png",
        rng_seed: int = 1234,
        random_flip: bool = False,
    ):
        """Dataset from a folder of images.

        Args:
            root (str): Path to images.
            img_size (int, optional): Image size. Defaults to 256.
            split (str, optional): "train", "valid", or "test".
            rng_seed (int, optional): Random seed. Defaults to 1234.
        """
        self.trans = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip() if random_flip else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        root = Path(root)
        if metafile is None:
            images = sorted(root.glob(f"**/*{img_ext}"))
        else:
            with open(metafile, "r") as reader:
                images = []
                for fn in reader.readlines():
                    file = root / (fn.strip() + img_ext)
                    if file.exists():
                        images.append(file)

        RandomState(rng_seed).shuffle(images)

        ds_size = len(images)
        lo, hi = int(ds_size * data_portion[0]), int(ds_size * data_portion[1])

        self.images = images[lo:hi]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]

        image = Image.open(file).convert('RGB')
        image = self.trans(image)

        return {"image": image}
