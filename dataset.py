import os 
import glob
import torch

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

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
    def __init__(self, image_ids, class_list, transform = True, preprocessing_fn = None, visual_evaluation = False, rgb = True) -> None:
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        self.class_list = class_list
        self.visual_evaluation = visual_evaluation
        self.rgb =rgb
        
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
        for i, image_id in enumerate(image_ids):
            try:
                files = glob.glob(os.path.join(TRAIN_PATH, image_id, "*.jpg")) # Check if image exist
            except:
                raise Exception(f"{os.path.join(TRAIN_PATH, image_id, '*.jpg')} does not exist")

            # Calculate mask path
            mask_paths = df[df["img_id"] == image_id]["mask_path"].values
            class_strs = df[df["img_id"] == image_id]["class"].values
            class_indexs = [self.class_list.index(class_str) for class_str in class_strs]

            self.data[i] = {
                "img_path": f"{os.path.join(TRAIN_PATH, image_id)}.jpg",
                "mask_paths": mask_paths, 
                "classes": class_indexs,
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item]["img_path"]
        mask_paths = self.data[item]["mask_paths"]
        classes = self.data[item]["classes"]

        # Image
        img = Image.open(image_path)
        if not self.rgb:
            img = ImageOps.grayscale(img)
        img = img.convert("RGB")
        img_original = img.copy()
        img = np.array(img)
        img = np.pad(img, ((2,2),(0,0), (0,0)), 'constant', constant_values = 0)
        
        # Mask
        mask = np.zeros((img.shape[0], img.shape[1]))
        
        for i, mask_path in enumerate(mask_paths):
            mask_part = Image.open(mask_path)
            mask_part = np.array(mask_part)
            mask_part = np.pad(mask_part, ((2,2),(0,0), (0,0)), 'constant', constant_values = self.class_list.index("unknown") + 1)
            
            mask_part = (mask_part >= 1).astype("float32") * (classes[i] + 1)
            mask_part = mask_part[:,:,0]

            mask_part = (mask == 0).astype("float32")*mask_part
            
            mask += mask_part

        zero_vals = (mask == 0).astype("float32") * (self.class_list.index("unknown") + 1)
        mask = zero_vals + mask -1


        # Transform if training data
        if self.transform is True:
            augmented = self.aug(image = img, mask = mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # preprocessing image
        img = self.preprocessing_fn(img)

        
        if self.visual_evaluation:
            return{
                "image" : transforms.ToTensor()(img),
                "mask" : transforms.ToTensor()(mask).float(),
                "image_to_show": img_original,
                }
        else:
            return{
                "image" : transforms.ToTensor()(img),
                "mask" : transforms.ToTensor()(mask).float(),
                }
     