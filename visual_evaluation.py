import torch
from PIL import Image, ImageFile
from dataset import SIIMDataset
import numpy as np
from matplotlib import cm, pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp
import pandas as pd
from matplotlib import colors, ticker
import matplotlib as mpl

MODEL_PATH = "trained_model_1.pt"

# define encoder for U-net
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
CLASSES_CSV = "./Labelbox/golf-examples/classes.csv"
TRAINING_CSV = "./Labelbox/golf-examples/images.csv"


df_classes = pd.read_csv(CLASSES_CSV, header=None)
class_list = list(df_classes.iloc[:,0].values)

# Model
model = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = len(class_list),
        activation  = "softmax"
    )
model_trained = torch.load(MODEL_PATH)
model.load_state_dict(model_trained.state_dict())
model.to('cpu')
model.eval()

# Data
df = pd.read_csv(TRAINING_CSV, header=None)
images = df.iloc[:, 0].values

prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

dataset = SIIMDataset(
    images,
    class_list,
    preprocessing_fn=prep_fn,
    transform = False,
)

i = 3

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

fig, axes = plt.subplots(nrows=2, ncols=2)

cmap = plt.get_cmap("Paired")
#cmap = colors.ListedColormap(['white', 'red', 'orange', 'black', 'grey','blue', 'yellow','green','pink', 'purple'])
bounds=range(11)
norm = colors.BoundaryNorm(bounds, cmap.N)

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

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cbar_ax) 
tick_locator = ticker.MaxNLocator(nbins=10)
cbar.locator = tick_locator
cbar.update_ticks()

class_list.insert(0, "")
cbar.ax.set_yticklabels(class_list) 

plt.show()
