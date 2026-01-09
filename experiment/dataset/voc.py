import copy
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def suppress_mask_idx(masks):
    """Make the mask index 0, 1, 2, ..."""
    # the original mask could have not continuous index, 0, 3, 4, 6, 9, 13, ...
    # we make them 0, 1, 2, 3, 4, 5, ...
    if isinstance(masks, np.ndarray):
        pkg = np
    elif isinstance(masks, torch.Tensor):
        pkg = torch
    else:
        raise NotImplementedError
    obj_idx = pkg.unique(masks)
    idx_mapping = pkg.arange(obj_idx.max() + 1)
    idx_mapping[obj_idx] = pkg.arange(len(obj_idx))
    masks = idx_mapping[masks]
    return masks


class RandomHorizontalFlip:
    """Flip the image and bbox horizontally."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        # [H, W, 3], [H, W(, 2)], [N, 5]
        image, masks = sample['image'], sample['masks']

        if np.random.uniform(0, 1) < self.prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            masks = np.ascontiguousarray(masks[:, ::-1])

        return {
            'image': image,
            'masks': masks,
        }


class ResizeMinShape:
    """Resize for later center crop."""

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        h, w, _ = image.shape
        # resize so that the h' is at lease resolution[0]
        # and the w' is at lease resolution[1]
        factor = max(self.resolution[0] / h, self.resolution[1] / w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(
            image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        masks = cv2.resize(
            masks, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        return {
            'image': image,
            'masks': masks,
        }


class CenterCrop:
    """Crop the center square of the image."""

    def __init__(self, resolution=(224, 224), random=False):
        self.resolution = resolution
        self.random = random  # if True, offset the center crop

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w, _ = image.shape
        assert h >= self.resolution[0] and w >= self.resolution[1]
        assert h == self.resolution[0] or w == self.resolution[1]

        if h == self.resolution[0]:
            crop_ymin = 0
            crop_ymax = h
            if self.random:
                crop_xmin = np.random.randint(0, w - self.resolution[1] + 1)
                crop_xmax = crop_xmin + self.resolution[1]
            else:
                crop_xmin = (w - self.resolution[0]) // 2
                crop_xmax = crop_xmin + self.resolution[0]
        else:
            crop_xmin = 0
            crop_xmax = w
            if self.random:
                crop_ymin = np.random.randint(0, h - self.resolution[1] + 1)
                crop_ymax = crop_ymin + self.resolution[1]
            else:
                crop_ymin = (h - self.resolution[1]) // 2
                crop_ymax = crop_ymin + self.resolution[1]
        image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        masks = masks[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        return {
            'image': image,
            'masks': masks,
        }


class Normalize:
    """Normalize the image with mean and std."""

    def __init__(self, mean=0.5, std=0.5):
        if isinstance(mean, (list, tuple)):
            mean = np.array(mean, dtype=np.float32)[None, None]  # [1, 1, 3]
        if isinstance(std, (list, tuple)):
            std = np.array(std, dtype=np.float32)[None, None]  # [1, 1, 3]
        self.mean = mean
        self.std = std

    def normalize_image(self, image):
        image = image.astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image

    def denormalize_image(self, image):
        # simple numbers
        if isinstance(self.mean, (int, float)) and \
                isinstance(self.std, (int, float)):
            image = image * self.std + self.mean
            return image.clamp(0, 1)
        # need to convert the shapes
        mean = image.new_tensor(self.mean.squeeze())  # [3]
        std = image.new_tensor(self.std.squeeze())  # [3]
        if image.shape[-1] == 3:  # C last
            mean = mean[None, None]  # [1, 1, 3]
            std = std[None, None]  # [1, 1, 3]
        else:  # C first
            mean = mean[:, None, None]  # [3, 1, 1]
            std = std[:, None, None]  # [3, 1, 1]
        if len(image.shape) == 4:  # [B, C, H, W] or [B, H, W, C], batch dim
            mean = mean[None]
            std = std[None]
        image = image * self.std + self.mean
        return image.clamp(0, 1)

    def __call__(self, sample):
        # [H, W, C]
        image, masks = sample['image'], sample['masks']
        image = self.normalize_image(image)
        # make mask index start from 0 and continuous
        # `masks` is [H, W(, 2 or 3)]
        if len(masks.shape) == 3:
            assert masks.shape[-1] in [2, 3]
            # we don't suppress the last mask since it is the overlapping mask
            # i.e. regions with overlapping instances
            for i in range(masks.shape[-1] - 1):
                masks[:, :, i] = suppress_mask_idx(masks[:, :, i])
        else:
            masks = suppress_mask_idx(masks)
        return {
            'image': image,
            'masks': masks,
        }


class VOCCollater:
    """Collect images, annotations, etc. into a batch."""

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['masks'] for s in data]

        images = np.stack(images, axis=0)  # [B, H, W, C]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [B, C, H, W]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)  # [B, H, W(, 2 or 3)]

        data_dict = {
            'image': images.contiguous().float(),
            'mask': masks.contiguous().long(),
        }

        if len(masks.shape) == 4:
            assert masks.shape[-1] in [2, 3]
            if masks.shape[-1] == 3:
                data_dict['mask'] = masks[:, :, :, 0]
                data_dict['sem_mask'] = masks[:, :, :, 1]
                data_dict['inst_overlap_mask'] = masks[:, :, :, 2]
            else:
                data_dict['mask'] = masks[:, :, :, 0]
                data_dict['inst_overlap_mask'] = masks[:, :, :, 1]

        return data_dict


class VOCTransforms(object):
    """Data pre-processing steps."""

    def __init__(
        self,
        resolution,
        norm_mean=0.5,
        norm_std=0.5,
        val=False,
    ):
        self.transforms = transforms.Compose([
            RandomHorizontalFlip(0.5 if not val else 0),
            ResizeMinShape(resolution),
            CenterCrop(resolution, random=(not val)),
            Normalize(norm_mean, norm_std),
        ])
        self.resolution = resolution

    def __call__(self, input):
        return self.transforms(input)


class VOC12Dataset(data.Dataset):

    VOC_CATEGORY_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]

    def __init__(
        self,
        data_root,
        split='val',
        img_size: int = 512,
        load_annotation=True,
        ignore_classes=[],
    ):
        # Set paths
        self.data_root = data_root
        valid_splits = ['trainaug', 'train', 'val']
        assert (split in valid_splits)
        self.split = split
        self.load_anno = load_annotation

        if split == 'trainaug':
            _semseg_dir = os.path.join(self.data_root, 'SegmentationClassAug')
        else:
            _semseg_dir = os.path.join(self.data_root, 'SegmentationClass')

        _instseg_dir = os.path.join(self.data_root, 'SegmentationObject')
        _image_dir = os.path.join(self.data_root, 'images')

        # Transform
        self.voc_transforms = VOCTransforms(
            (img_size, img_size),
            norm_mean=0.5,
            norm_std=0.5,
            val=split == "val",
        )

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(
            self.split)))
        split_file = os.path.join(self.data_root, 'sets', self.split + '.txt')
        self.images, self.semsegs, self.instsegs = [], [], []

        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            # Instance Segmentation
            # only available for val set
            if self.split == 'val':
                _instseg = os.path.join(_instseg_dir, line + '.png')
                assert os.path.isfile(_instseg)
            else:
                _instseg = _semseg
            self.instsegs.append(_instseg)

        assert (len(self.images) == len(self.semsegs) == len(self.instsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        self.ignore_classes = [
            self.VOC_CATEGORY_NAMES.index(class_name)
            for class_name in ignore_classes
        ]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)  # [H, W, 3]
        sample['image'] = _img
        if not self.load_anno:
            H, W = _img.shape[:2]
            sample['masks'] = np.zeros((H, W), dtype=np.int32)
            return self.voc_transforms(sample)

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)  # [H, W]
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(
                _semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        inst_overlap_masks = (_semseg == 255)

        # Load inst_seg mask
        # we don't have it for training set! Only load for val set
        if self.split == 'val':
            _instseg = self._load_instseg(index)  # [H, W]
            inst_overlap_masks = (_instseg == 255) | inst_overlap_masks
        else:
            _instseg = copy.deepcopy(_semseg)  # [H, W], fake it
        _semseg[inst_overlap_masks] = 0
        _instseg[inst_overlap_masks] = 0
        inst_overlap_masks = inst_overlap_masks.astype(np.uint8)
        masks = [_instseg, _semseg, inst_overlap_masks]
        sample['masks'] = np.stack(masks, axis=-1)  # [H, W, 3]

        return self.voc_transforms(sample)

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img.astype(np.uint8)

    def _load_semseg(self, index):
        # background is 0, 255 is ignored
        _semseg = np.array(Image.open(self.semsegs[index]))  # [H, W]
        for ignore_class in self.ignore_classes:
            _semseg[_semseg == ignore_class] = 255
        return _semseg

    def _load_instseg(self, index):
        # background is 0, 255 is ignored
        _instseg = np.array(Image.open(self.instsegs[index]))  # [H, W]
        return _instseg

    def collate_fn(self):
        return VOCCollater()


def build_voc_dataset(params, val_only=False):
    """Build VOC12 dataset that load images."""
    norm_mean = params.get('norm_mean', 0.5)
    norm_std = params.get('norm_std', 0.5)
    val_transforms = VOCTransforms(
        params.resolution,
        norm_mean=norm_mean,
        norm_std=norm_std,
        val=True,
    )
    args = dict(
        data_root=params.data_root,
        voc_transforms=val_transforms,
        split='val',
        load_anno=params.load_anno,
    )
    val_dataset = VOC12Dataset(**args)
    if val_only:
        return val_dataset, VOCCollater()
    args['split'] = 'trainaug'
    args['load_anno'] = False
    args['voc_transforms'] = VOCTransforms(
        params.resolution,
        norm_mean=norm_mean,
        norm_std=norm_std,
        val=False,
    )
    train_dataset = VOC12Dataset(**args)
    return train_dataset, val_dataset, VOCCollater()
