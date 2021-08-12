from matplotlib import pyplot as plt
import numpy as np




file_name = "train_pixel_accuracy_1.npy"
train_pixel_acc = np.load(file_name)

file_name = "validation_pixel_accuracy_1.npy"
val_pixel_acc = np.load(file_name)

file_name = "train_miou_accuracy_1.npy"
train_miou_acc = np.load(file_name)

file_name = "validation_miou_accuracy_1.npy"
val_miou_acc = np.load(file_name)

file_name = "validation_loss_1.npy"
val_loss = np.load(file_name)

plt.figure()
plt.plot(train_pixel_acc, label="Train Pixel")
plt.plot(val_pixel_acc, label="Validation Pixel")
plt.legend()

plt.figure()
plt.plot(train_miou_acc, label="Train mIoU")
plt.plot(val_miou_acc, label="Validation mIoU")
plt.legend()

plt.figure()
plt.plot(val_loss)

plt.show()