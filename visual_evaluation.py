import torch
from PIL import Image, ImageFile
from dataset import SIIMDataset
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp
import pandas as pd

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
)



""" image_path = "Labelbox/golf-examples/images/ckrlpgdth3lwm0ypfd04k1rxw.jpg"
img = Image.open(image_path)
img = img.convert("RGB")
img = np.array(img)
img = np.pad(img, ((2,2),(0,0), (0,0)), 'constant', constant_values = 0)
img = prep_fn(img)
img = transforms.ToTensor()(img) """
data = dataset[3]
img = data["image"]
img = img.to('cpu', dtype=torch.float)

with torch.no_grad():
    img.unsqueeze(0)
    output = model(img)

plt.imshow(output)
