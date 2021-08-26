import os
import torch
from dataset import SIIMDataset
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import pandas as pd
from matplotlib import colors, ticker
import matplotlib as mpl

INSTANCE = "6"
RESULT_PATH = os.path.join("result", INSTANCE)

MODEL_PATH = os.path.join(RESULT_PATH, f"{INSTANCE}_trained_model.pt")

# define encoder for U-net
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"

# data to test on
DATASET = "golf-2"
CLASSES_CSV = os.path.join("Labelbox", DATASET, "classes.csv")
TRAINING_CSV = os.path.join("Labelbox", DATASET, "images.csv")

#TEST_DATASET_FILE = None
TEST_DATASET_FILE = os.path.join(RESULT_PATH, f"{INSTANCE}_test_dataset.pt")


df_classes = pd.read_csv(CLASSES_CSV, header=None)
class_list = list(df_classes.iloc[:,0].values)

# Model
model = torch.load(MODEL_PATH)
model.to('cpu')
model.eval()

# Data
if not TEST_DATASET_FILE:
    df = pd.read_csv(TRAINING_CSV, header=None)
    images = df.iloc[:, 0].values

    prep_fn = smp.encoders.get_preprocessing_fn(
            ENCODER,
            ENCODER_WEIGHTS
        )

    dataset = SIIMDataset(
        images,
        class_list,
        DATASET,
        preprocessing_fn=prep_fn,
        transform = False,
        visual_evaluation=True,
        rgb = False
    )
else:
    dataset = torch.load(TEST_DATASET_FILE)

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,13))
cmap = plt.get_cmap("Paired")
bounds=range(11)
norm = colors.BoundaryNorm(bounds, cmap.N)

for i in range(len(dataset)):
    print(f"waiting for {i}")
    if not plt.waitforbuttonpress():
        print(i)

        # Image
        data = dataset[i]
        img = data["image"]

        img_img = data["image_to_show"]
        img = img.to('cpu', dtype=torch.float)

        # Predict
        with torch.no_grad():
            
            img = img.unsqueeze(0)
            output = model(img)

        output = output[0,:,:,:]
        output = torch.argmax(output, 0)

        # Mask
        mask = data["mask"]
        mask = mask[0,:,:]

        # Plot

        axes[0,0].imshow(img_img)
        axes[0,0].imshow(output, alpha=0.6, cmap=cmap, norm=norm)
        axes[0,0].set_title("Predicted")

        axes[0,1].imshow(output, cmap=cmap, norm=norm)
        axes[0,1].set_title("Predicted mask")

        axes[1,0].imshow(img_img)
        axes[1,0].imshow(mask, alpha=0.6, cmap=cmap, norm=norm)
        axes[1,0].set_title("Original")

        pos = axes[1,1].imshow(mask, cmap=cmap, norm=norm)
        axes[1,1].set_title("Original mask")

        # Cbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax) 
        tick_locator = ticker.MaxNLocator(nbins=10)
        cbar.locator = tick_locator
        cbar.update_ticks()

        class_list = list(df_classes.iloc[:,0].values)
        if len(class_list) < 10:
            for i in range(10-len(class_list)):
                class_list.append("-")
        class_list.insert(0, "")
        cbar.ax.set_yticklabels(class_list) 
        
        plt.draw()
