import os
from matplotlib import pyplot as plt
import numpy as np


INSTANCE = "6"

RESULT_PATH = os.path.join("result", INSTANCE)

train_pixel_acc = np.load(os.path.join(RESULT_PATH, f"{INSTANCE}_train_pixel_accuracy.npy"))
val_pixel_acc = np.load(os.path.join(RESULT_PATH, f"{INSTANCE}_validation_pixel_accuracy.npy"))
train_miou_acc = np.load(os.path.join(RESULT_PATH, f"{INSTANCE}_train_miou_accuracy.npy"))
val_miou_acc = np.load(os.path.join(RESULT_PATH, f"{INSTANCE}_validation_miou_accuracy.npy"))
val_loss = np.load(os.path.join(RESULT_PATH, f"{INSTANCE}_validation_loss.npy"))


plt.plot(train_pixel_acc, label="Train Pixel")
plt.plot(val_pixel_acc, label="Validation Pixel")
plt.legend()
plt.title("Pixel accuracy")

plt.savefig(os.path.join(RESULT_PATH, "pixel_accuracy.png"))
plt.show()


plt.plot(train_miou_acc, label="Train mIoU")
plt.plot(val_miou_acc, label="Validation mIoU")
plt.legend()
plt.title("mIoU accuracy")

plt.savefig(os.path.join(RESULT_PATH, "mIoU_accuracy.png"))
plt.show()


plt.plot(val_loss)
plt.title("Validation loss")

plt.savefig(os.path.join(RESULT_PATH, "validation_loss.png"))
plt.show()

