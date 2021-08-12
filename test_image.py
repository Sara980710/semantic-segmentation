import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from matplotlib import pyplot as plt


MASK_CSV_PATH = "./Labelbox/golf-examples/annotations_seg.csv"
TRAINING_CSV = "./Labelbox/golf-examples/images.csv"
CLASSES_CSV = "./Labelbox/golf-examples/classes.csv"

df = pd.read_csv(TRAINING_CSV, header=None)
df.columns = ["img_id", "img_path"]

df_classes = pd.read_csv(CLASSES_CSV, header=None)
class_list = list(df_classes.iloc[:,0].values)


for i in range(100):
    image_id = df["img_id"].values[i]
    image_path = df["img_path"].values[i]

    img = Image.open(image_path)

    df_mask = pd.read_csv(MASK_CSV_PATH, header=None)
    df_mask.columns = ["mask_path", "img_id", "class", "A", "B", "C", "D"]

    mask_paths = df_mask[df_mask["img_id"] == image_id]["mask_path"].values
    class_strs = df_mask[df_mask["img_id"] == image_id]["class"].values
    class_indexs = [class_list.index(class_str) for class_str in class_strs]
    mask = np.zeros((np.array(img).shape[0]+4, np.array(img).shape[1]))
    for i, mask_path in enumerate(mask_paths):
        mask_part = Image.open(mask_path)
        mask_part = np.array(mask_part)
        mask_part = np.pad(mask_part, ((2,2),(0,0), (0,0)), 'constant', constant_values = class_list.index("unknown") + 1)
        
        mask_part = (mask_part >= 1).astype("float32") * (class_indexs[i] + 1)
        mask_part = mask_part[:,:,0]

        mask_part = (mask == 0).astype("float32")*mask_part
        
        mask += mask_part

    zero_vals = (mask == 0).astype("float32") * (class_list.index("unknown") + 1)
    mask = (zero_vals + mask -1)*100


    width, height = img.size
    left = 0
    top = 140 + 16*2
    right = width
    bottom = height
    img = img.crop((left, top, right, bottom))
    mask = mask[:][top:]
    print(f"size = {img.size}")

    plt.figure()
    plt.imshow(img)
    
    plt.figure()
    plt.imshow(mask)
    plt.show()