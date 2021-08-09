import os 
import glob
import torch

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms

from albumentations import (
    Compose, 
    OneOf, 
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAIN_PATH = "./Labelbox/golf-examples/images"
MASK_CSV_PATH = "./Labelbox/golf-examples/annotations_seg.csv"

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, transform = True, preprocessing_fn = None) -> None:
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit = 0.0625,
                    scale_limit = 0.1,
                    rotate_limit = 10, 
                    p = 0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit = (90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit = 0.1, 
                            contrast_limit = 0.1
                        ),
                    ],
                    p = 0.5,
                ),
            ]
        )

        df = pd.read_csv(MASK_CSV_PATH)
        df.columns = ["mask_path", "img_id", "class", "A", "B", "C", "D"]
        for i, image_id in enumerate(image_ids):
            files = glob.glob(os.path.join(TRAIN_PATH, image_id, "*.jpg")) # Check if image exist

            # Calculate mask path
            mask_paths = df[df["img_id"] == image_id]["mask_path"].values

            self.data[i] = {
                "img_path": f"{os.path.join(TRAIN_PATH, image_id)}.jpg",
                "mask_path": mask_paths
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]

        img = Image.open(image_path)
        img = img.convert("RGB")
        img = np.array(img)
        img = np.pad(img, ((2,2),(0,0), (0,0)), 'constant', constant_values = 0)

        mask = Image.open(mask_path[0])
        mask = np.array(mask)
        mask = np.pad(mask, ((2,2),(0,0), (0,0)), 'constant', constant_values = 0)
        mask = (mask >= 1).astype("float32")


        # Transform if training data
        if self.transform is True:
            augmented = self.aug(image = img, mask = mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # preprocessing image
        img = self.preprocessing_fn(img)

        mask = mask[:,:,0]

        return{
            "image" : transforms.ToTensor()(img),
            "mask" : transforms.ToTensor()(mask).float(),
            }
     