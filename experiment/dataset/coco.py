# https://github.com/Wuziyi616/SlotDiffusion/blob/main/slotdiffusion/img_based/datasets/coco.py

import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms as T


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
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']

        if np.random.uniform(0, 1) < self.prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            masks = np.ascontiguousarray(masks[:, ::-1])
            _, w, _ = image.shape
            # adjust annos
            if annos.shape[0] > 0:
                x1 = annos[:, 0].copy()
                x2 = annos[:, 2].copy()
                annos[:, 0] = w - x2
                annos[:, 2] = w - x1

        return {
            'image': image,
            'masks': masks,
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class ResizeMinShape:
    """Resize for later center crop."""

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def __call__(self, sample):
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']
        h, w, _ = image.shape
        # resize so that the h' is at lease resolution[0]
        # and the w' is at lease resolution[1]
        factor = max(self.resolution[0] / h, self.resolution[1] / w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(
            image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        masks = cv2.resize(
            masks, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        # adjust annos
        factor = float(factor)
        annos[:, :4] *= factor
        scale *= factor
        return {
            'image': image,
            'masks': masks,
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class CenterCrop:
    """Crop the center square of the image."""

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def __call__(self, sample):
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']

        h, w, _ = image.shape
        assert h >= self.resolution[0] and w >= self.resolution[1]
        assert h == self.resolution[0] or w == self.resolution[1]

        if h == self.resolution[0]:
            crop_ymin = 0
            crop_ymax = h
            crop_xmin = (w - self.resolution[0]) // 2
            crop_xmax = crop_xmin + self.resolution[0]
        else:
            crop_xmin = 0
            crop_xmax = w
            crop_ymin = (h - self.resolution[1]) // 2
            crop_ymax = crop_ymin + self.resolution[1]
        image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        masks = masks[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        # adjust annos
        if annos.shape[0] > 0:
            annos[:, [0, 2]] = annos[:, [0, 2]] - crop_xmin
            annos[:, [1, 3]] = annos[:, [1, 3]] - crop_ymin
            # filter out annos that are out of the image
            mask1 = (annos[:, 2] > 0) & (annos[:, 3] > 0)
            mask2 = (annos[:, 0] < self.resolution[0]) & \
                (annos[:, 1] < self.resolution[1])
            annos = annos[mask1 & mask2]
            annos[:, [0, 2]] = np.clip(annos[:, [0, 2]], 0, self.resolution[0])
            annos[:, [1, 3]] = np.clip(annos[:, [1, 3]], 0, self.resolution[1])

        return {
            'image': image,
            'masks': masks,
            'annos': annos,
            'scale': scale,
            'size': size,
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
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']
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
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class COCOCollater:
    """Collect images, annotations, etc. into a batch."""

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['masks'] for s in data]
        annos = [s['annos'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        images = np.stack(images, axis=0)  # [B, H, W, C]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [B, C, H, W]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)  # [B, H, W(, 2 or 3)]

        max_annos_num = max(anno.shape[0] for anno in annos)
        if max_annos_num > 0:
            input_annos = np.ones(
                (len(annos), max_annos_num, 5), dtype=np.float32) * (-1)
            for i, anno in enumerate(annos):
                if anno.shape[0] > 0:
                    input_annos[i, :anno.shape[0], :] = anno
        else:
            input_annos = np.ones((len(annos), 1, 5), dtype=np.float32) * (-1)
        input_annos = torch.from_numpy(input_annos).float()

        scales = torch.from_numpy(np.array(scales)).float()
        sizes = torch.from_numpy(np.array(sizes)).float()

        data_dict = {
            'image': images.contiguous().float(),
            'mask': masks.contiguous().long(),
            'annotation': input_annos,
            'scale': scales,
            'size': sizes,
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


class COCOTransforms(object):
    """Data pre-processing steps."""

    def __init__(
        self,
        resolution,
        norm_mean=0.5,
        norm_std=0.5,
        val=False,
    ):
        self.normalize = Normalize(norm_mean, norm_std)
        self.transforms = transforms.Compose([
            RandomHorizontalFlip(0.5 if not val else 0),
            ResizeMinShape(resolution),
            CenterCrop(resolution),
            self.normalize,
        ])
        self.resolution = resolution

    def __call__(self, input):
        return self.transforms(input)


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

COCO_CLASSES_COLOR = [
    (241, 23, 78), (63, 71, 49), (67, 79, 143),
    (32, 250, 205), (136, 228, 157), (135, 125, 104),
    (151, 46, 171), (129, 37, 28), (3, 248, 159),
    (154, 129, 58), (93, 155, 200), (201, 98, 152),
    (187, 194, 70), (122, 144, 121), (168, 31, 32),
    (168, 68, 189), (173, 68, 45), (200, 81, 154),
    (171, 114, 139), (216, 211, 39), (187, 119, 238),
    (201, 120, 112), (129, 16, 164), (211, 3, 208),
    (169, 41, 248), (100, 77, 159), (140, 104, 243),
    (26, 165, 41), (225, 176, 197), (35, 212, 67),
    (160, 245, 68), (7, 87, 70), (52, 107, 85),
    (103, 64, 188), (245, 76, 17), (248, 154, 59),
    (77, 45, 123), (210, 95, 230), (172, 188, 171),
    (250, 44, 233), (161, 71, 46), (144, 14, 134),
    (231, 142, 186), (34, 1, 200), (144, 42, 108),
    (222, 70, 139), (138, 62, 77), (178, 99, 61),
    (17, 94, 132), (93, 248, 254), (244, 116, 204),
    (138, 165, 238), (44, 216, 225), (224, 164, 12),
    (91, 126, 184), (116, 254, 49), (70, 250, 105),
    (252, 237, 54), (196, 136, 21), (234, 13, 149),
    (66, 43, 47), (2, 73, 234), (118, 181, 5),
    (105, 99, 225), (150, 253, 92), (59, 2, 121),
    (176, 190, 223), (91, 62, 47), (198, 124, 140),
    (100, 135, 185), (20, 207, 98), (216, 38, 133),
    (17, 202, 208), (216, 135, 81), (212, 203, 33),
    (108, 135, 76), (28, 47, 170), (142, 128, 121),
    (23, 161, 179), (33, 183, 224)
]


def to_rgb_from_tensor(x):
    """Reverse the Normalize operation in torchvision."""
    return (x * 0.5 + 0.5).clamp(0, 1)


def _draw_bbox(img, anno, bbox_width=2):
    """Draw bbox on images.

    Args:
        img: (3, H, W), torch.Tensor
        anno: (N, 5)
    """
    anno = anno[anno[:, -1] != -1]
    img = torch.round((to_rgb_from_tensor(img) * 255.)).to(dtype=torch.uint8)
    bbox = anno[:, :4]
    label = anno[:, -1]
    draw_label = [COCO_CLASSES[int(lbl)] for lbl in label]
    draw_color = [COCO_CLASSES_COLOR[int(lbl)] for lbl in label]
    bbox_img = vutils.draw_bounding_boxes(
        img, bbox, labels=draw_label, colors=draw_color, width=bbox_width)
    bbox_img = bbox_img.float() / 255. * 2. - 1.
    return bbox_img


def draw_coco_bbox(imgs, annos, bbox_width=2):
    """Draw bbox on batch images.

    Args:
        imgs: (B, 3, H, W), torch.Tensor
        annos: (B, N, 5)
    """
    if len(imgs.shape) == 3:
        return draw_coco_bbox(imgs[None], annos[None], bbox_width)[0]

    bbox_imgs = []
    for img, anno in zip(imgs, annos):
        bbox_imgs.append(_draw_bbox(img, anno, bbox_width=bbox_width))
    bbox_imgs = torch.stack(bbox_imgs, dim=0)
    return bbox_imgs


class COCO2017Dataset(Dataset):
    """COCO 2017 dataset."""

    def __init__(
        self,
        data_root,
        split: str = "train",
        img_size: int = 512,
        load_annotation: bool = True,
        load_sem_mask: bool = True
    ):
        set_name = f'{split}2017'
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.split = split
        self.load_annotation = load_annotation

        self.image_dir = os.path.join(data_root, 'images', set_name)
        self.anno_dir = os.path.join(data_root, 'annotations',
                                     f'instances_{set_name}.json')
        self.coco = COCO(self.anno_dir)

        self.image_ids = self.coco.getImgIds()

        if split == 'train':
            # filter image id without annotation
            ids = []
            for image_id in self.image_ids:
                anno_ids = self.coco.getAnnIds(imgIds=image_id)
                annos = self.coco.loadAnns(anno_ids)
                if len(annos) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids

        self.cat_ids = self.coco.getCatIds()
        self.cats = sorted(
            self.coco.loadCats(self.cat_ids), key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 79
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}
        self.cat_id_to_coco_label = {
            cat['id']: i
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_id = {
            i: cat['id']
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_name = {
            coco_label: self.cat_id_to_cat_name[cat_id]
            for coco_label, cat_id in self.coco_label_to_cat_id.items()
        }

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

        # by default only load instance seg_mask, not semantic seg_mask
        self.load_sem_mask = load_sem_mask

        self.trans_en = T.Compose([
            ResizeMinShape((img_size, img_size)),
            CenterCrop((img_size, img_size)),
            Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        H, W = image.shape[:2]

        if self.load_annotation:
            annos = self.load_annos(idx)  # [N, 5]
            masks, inst_overlap_masks = self.load_inst_masks(idx)  # [H, W]x2
            masks = [masks, inst_overlap_masks]
            if self.load_sem_mask:
                sem_masks = self.load_sem_masks(idx)  # [H, W]
                masks.insert(1, sem_masks)  # [inst, sem, inst_overlap]
            masks = np.stack(masks, axis=-1)  # [H, W, 2 or 3]
        else:
            annos = np.zeros((0, 5), dtype=np.float32)
            masks = np.zeros((H, W), dtype=np.int32)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = self.trans_en(
            {
                'image': image,
                'masks': masks,
                'annos': annos,
                'scale': scale,
                'size': size,
            }
        )

        return sample

    def load_image(self, idx):
        """Read image."""
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        image = cv2.imdecode(
            np.fromfile(
                os.path.join(self.image_dir, file_name), dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.uint8)

    def load_annos(self, idx):
        """Load bbox and cls."""
        anno_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annos = self.coco.loadAnns(anno_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        targets = np.zeros((0, 5))
        if len(annos) == 0:
            return targets.astype(np.float32)

        # filter annos
        for anno in annos:
            if anno.get('ignore', False):
                continue
            if anno.get('iscrowd', False):
                continue
            if anno['category_id'] not in self.cat_ids:
                continue

            # bbox format: [x_min, y_min, w, h]
            bbox = anno['bbox']
            inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
            inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
            if inter_w * inter_h == 0:
                continue
            if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                continue

            target = np.zeros((1, 5))
            target[0, :4] = bbox
            target[0, 4] = self.cat_id_to_coco_label[anno['category_id']]
            targets = np.append(targets, target, axis=0)

        # [x_min, y_min, w, h] --> [x_min, y_min, x_max, y_max]
        targets[:, 2] = targets[:, 0] + targets[:, 2]
        targets[:, 3] = targets[:, 1] + targets[:, 3]

        return targets.astype(np.float32)  # [N, 5 (x1, y1, x2, y2, cat_id)]

    def load_inst_masks(self, idx):
        """Load instance seg_mask and merge them into an argmaxed mask."""
        anno_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annos = self.coco.loadAnns(anno_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        masks = np.zeros((image_h, image_w), dtype=np.int32)
        inst_overlap_masks = np.zeros_like(masks)  # for overlap check
        for i, anno in enumerate(annos):
            if anno.get('ignore', False):
                continue
            if anno.get('iscrowd', False):
                continue
            if anno['category_id'] not in self.cat_ids:
                continue
            mask = self.coco.annToMask(anno)
            masks[mask > 0] = i + 1  # to put background as 0
            inst_overlap_masks[mask > 0] += 1
        # overlap value > 1 indicates overlap
        inst_overlap_masks = (inst_overlap_masks > 1).astype(np.int32)
        # [H, W], [H, W], 1 is overlapping pixels
        return masks, inst_overlap_masks

    def load_sem_masks(self, idx):
        """Load instance seg_mask and merge them into an argmaxed mask."""
        anno_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annos = self.coco.loadAnns(anno_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        masks = np.zeros((image_h, image_w), dtype=np.int32)
        for i, anno in enumerate(annos):
            if anno.get('ignore', False):
                continue
            if anno.get('iscrowd', False):
                continue
            if anno['category_id'] not in self.cat_ids:
                continue
            mask = self.coco.annToMask(anno)
            coco_lbl = self.cat_id_to_coco_label[anno['category_id']]
            masks[mask > 0] = coco_lbl + 1  # to put background as 0
        # [H, W]
        return masks

    def collate_fn(self):
        return COCOCollater()
