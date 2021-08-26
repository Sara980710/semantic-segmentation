import os
import torch

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

#from apex import amp
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset

# training csv file path
DATASET = "golf-3"
TRAINING_CSV = os.path.join("Labelbox", DATASET, "images.csv")
CLASSES_CSV = os.path.join("Labelbox", DATASET, "classes.csv")
INSTANCE = "7" # For saving files

# training and test batch sizes
TRAINING_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

# If no test set has been recorded
MAKE_TEST_SET = False

# number of workers
WORKERS = 4

# number of epochs
EPOCHS = 5000

# define encoder for U-net
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"

DEVICE = "cpu"

def pixel_accuracy(output, mask):
    output = torch.argmax(F.softmax(output, dim=1), dim=1)
    correct = torch.eq(output, mask).int()
    accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def train(dataset, data_loader, model, criterion, optimizer, n_classes):
    # Put model in training mode
    model.train()

    # Calc number of batches
    num_batches = int(len(dataset) / data_loader.batch_size)

    # Init tqdm
    tk0 = tqdm(data_loader, total=num_batches)

    # Accuracies
    accuracy = 0
    miou_accuracy = 0

    # Loop over all batches
    for i, d in enumerate(tk0):
        inputs = d["image"]
        targets = d["mask"]

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        targets = torch.reshape(targets, (targets.size()[0], 544, 960))

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            accuracy +=  pixel_accuracy(outputs, targets)
            miou_accuracy += mIoU(outputs, targets, smooth=1e-10, n_classes=n_classes)
    
    tk0.close()
    return accuracy / num_batches, miou_accuracy / num_batches

def evaluate(dataset, data_loader, model, n_classes):
    model.eval()
    final_loss = 0
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    accuracy = 0
    miou_accuracy = 0

    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.long)
            output = model(inputs)

            targets = torch.reshape(targets, (targets.size()[0], 544, 960))
            loss = criterion(output, targets)
            final_loss += loss
            accuracy +=  pixel_accuracy(output, targets)
            miou_accuracy += mIoU(output, targets, smooth=1e-10, n_classes=n_classes)

    tk0.close()

    return final_loss / num_batches, accuracy / num_batches, miou_accuracy / num_batches



if __name__ == "__main__":
    df_classes = pd.read_csv(CLASSES_CSV, header=None)
    class_list = list(df_classes.iloc[:,0].values)

    df = pd.read_csv(TRAINING_CSV, header=None)
    df_train, df_val = model_selection.train_test_split(df, random_state=42, test_size=0.2)
    if MAKE_TEST_SET:
        df_train, df_test = model_selection.train_test_split(df_train, random_state=42, test_size=0.2)

    training_images = df_train.iloc[:, 0].values
    validation_images = df_val.iloc[:, 0].values
    if MAKE_TEST_SET:
        test_images = df_test.iloc[:, 0].values

    model = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = len(class_list),
        activation="softmax"
    )

    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    model.to(DEVICE)
    train_dataset = SIIMDataset(
        training_images,
        class_list,
        DATASET,
        preprocessing_fn=prep_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = WORKERS
    )

    val_dataset = SIIMDataset(
        validation_images,
        class_list,
        DATASET,
        transform=False,
        preprocessing_fn=prep_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = True,
        num_workers = WORKERS
    )

    if MAKE_TEST_SET:
        test_dataset = SIIMDataset(
            test_images,
            class_list,
            DATASET,
            preprocessing_fn=prep_fn,
            transform = False,
            visual_evaluation=True,
            rgb = False
        )
    RESULT_PATH = os.path.join("result", INSTANCE)
    os.mkdir(RESULT_PATH)
    if MAKE_TEST_SET:
        torch.save(test_dataset, os.path.join(RESULT_PATH, f"{INSTANCE}_test_dataset.pt"))
    torch.save(train_dataset, os.path.join(RESULT_PATH, f"{INSTANCE}_train_dataset.pt"))
    torch.save(val_dataset, os.path.join(RESULT_PATH, f"{INSTANCE}_validation_dataset.pt"))
    f = open(os.path.join(RESULT_PATH, f"{INSTANCE}_info.txt"), "a")
    f.write(f"dataset:{DATASET}")
    f.close()

    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1e-3
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=False
    )

    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Validation batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Num training images: {len(train_dataset)}")
    print(f"Num validation images: {len(val_dataset)}")
    if MAKE_TEST_SET:
        print(f"Num test images: {len(test_dataset)}")
    print(f"Encoder: {ENCODER}")
    print(f"Encoder weights: {ENCODER_WEIGHTS}")
    print("\n")

    train_pixel_acc_list = []
    train_miou_acc_list = []
    val_loss_list = []
    val_pixel_acc_list = []
    val_miou_acc_list = []

    for epoch in range(EPOCHS):
        # Training
        print(f"Training epoch: {epoch}")
        pixel_acc, miou_acc = train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer, 
            len(class_list)
        )
        print(f"Pixel accuracy: {pixel_acc}")
        print(f"mIou accuracy: {miou_acc}")

        # Validation
        print(f"Validation Epoch: {epoch}")
        val_log, val_pixel_acc, val_miou_acc = evaluate(
            val_dataset, val_loader, model, len(class_list)
        )
        scheduler.step(val_log)
        print(f"Validation loss: {val_log}")
        print(f"Pixel accuracy: {val_pixel_acc}")
        print(f"mIou accuracy: {val_miou_acc}")
        print("\n")

        # Append to lists
        train_pixel_acc_list.append(pixel_acc)
        train_miou_acc_list.append(miou_acc)
        val_loss_list.append(val_log)
        val_pixel_acc_list.append(val_pixel_acc)
        val_miou_acc_list.append(val_miou_acc)

        np.save(os.path.join(RESULT_PATH, f"{INSTANCE}_train_pixel_accuracy.npy"), train_pixel_acc_list)
        np.save(os.path.join(RESULT_PATH, f"{INSTANCE}_train_miou_accuracy.npy"), train_miou_acc_list)
        np.save(os.path.join(RESULT_PATH, f"{INSTANCE}_validation_loss.npy"), val_loss_list)
        np.save(os.path.join(RESULT_PATH, f"{INSTANCE}_validation_pixel_accuracy.npy"), val_pixel_acc_list)
        np.save(os.path.join(RESULT_PATH, f"{INSTANCE}_validation_miou_accuracy.npy"), val_miou_acc_list)

        torch.save(model, os.path.join(RESULT_PATH, f"{INSTANCE}_trained_model.pt"))
        

    print("WOHOO, Training is don!!!!")

    