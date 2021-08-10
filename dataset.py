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
    def __init__(self, image_ids, class_list, transform = True, preprocessing_fn = None) -> None:
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        self.class_list = class_list
        
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

        df = pd.read_csv(MASK_CSV_PATH, header=None)
        df.columns = ["mask_path", "img_id", "class", "A", "B", "C", "D"]
        counter = 0
        for index, row in df.iterrows():
            #files = glob.glob(os.path.join(TRAIN_PATH, image_id, "*.jpg")) # Check if image exist
            if row['img_id'] in image_ids:
                self.data[counter] = {
                    "img_path": f"{os.path.join(TRAIN_PATH, row['img_id'])}.jpg",
                    "mask_path": row['mask_path'], 
                    "class": row['class'], 
                }
                counter += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]
        class_index = self.class_list.index(self.data[item]["class"])

        img = Image.open(image_path)
        img = img.convert("RGB")
        img = np.array(img)
        img = np.pad(img, ((2,2),(0,0), (0,0)), 'constant', constant_values = 0)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.pad(mask, ((2,2),(0,0), (0,0)), 'constant', constant_values = 0)
        mask = (mask >= 1).astype("float32") * class_index


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
     