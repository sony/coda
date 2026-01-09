# https://github.com/Wuziyi616/SlotDiffusion/blob/86b302cd8dcdacaaca2ad0e41e36ba34a3ca3037/scripts/data_utils/download_movi.py


import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Example usage
install("tensorflow")
install("tensorflow_datasets")

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import tensorflow_datasets as tfds
from torchvision import transforms
import torchvision.utils as vutils
import tensorflow as tf
import json

parser = argparse.ArgumentParser()

parser.add_argument('--out_path', default='./data/MOVi/')
parser.add_argument('--level', default='e')
parser.add_argument('--image_size', type=int, default=256)

args = parser.parse_args()

ds, ds_info = tfds.load(
    f"movi_{args.level}/{args.image_size}x{args.image_size}:1.0.0",
    data_dir="gs://kubric-public/tfds",
    with_info=True)

to_tensor = transforms.ToTensor()

class JsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.RaggedTensor):
            return obj.to_tensor().numpy().tolist()
        elif isinstance(obj, bytes):
            return obj.decode()

        return json.JSONEncoder.default(self, obj)

def save_one_split(split):
    b = 0
    all_paths = []
    data_iter = iter(tfds.as_numpy(ds[split]))
    for record in tqdm(data_iter):
        video = record['video']
        masks = record['segmentations']
        
        instances = {}
        for k, v in record['instances'].items():
            if isinstance(v, tf.RaggedTensor):
                v = v.to_tensor().numpy()
            instances[k] = v
        
        T, *_ = video.shape
        assert masks.shape[0] == T

        # setup dirs
        path_vid = os.path.join(args.out_path, split, f"{b:08}")
        os.makedirs(path_vid, exist_ok=True)

        for t in range(T):
            img = video[t]
            img = to_tensor(img)
            vutils.save_image(img, os.path.join(path_vid, f"{t:08}.jpg"))

            mask = masks[t, ..., 0].astype(np.uint8)
            cv2.imwrite(os.path.join(path_vid, f"{t:08}_mask.png"), mask)

            instances_t = {}
            for k, v in instances.items():
                if len(v.shape) > 1 and v.shape[1] == T:
                    instances_t[k] = v[:, t]
                else:
                    instances_t[k] = v
            with open(os.path.join(path_vid, f"{t:08}_instances.json"), 'w') as f:
                json.dump(instances_t, f, cls=JsonEncoder)

        b += 1
        all_paths.append(os.path.dirname(path_vid))

    return all_paths


save_one_split('train')
save_one_split('validation')
save_one_split('test')