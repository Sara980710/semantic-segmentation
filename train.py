import os
import sys
from typing import final
from pandas.core.algorithms import mode
import torch

import numpy as np
import pandas as pd
from torch._C import dtype
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.nn.modules import activation
import torch.optim as optim

#from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset

from matplotlib import pyplot as plt

# training csv file path
TRAINING_CSV = "./Labelbox/golf-examples/images.csv"
CLASSES_CSV = "./Labelbox/golf-examples/classes.csv"

# training and test batch sizes
TRAINING_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

# number of epochs
EPOCHS = 10

# define encoder for U-net
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"

DEVICE = "cpu"

def train(dataset, data_loader, model, criterion, optimizer):
    # Put model in training mode
    model.train()

    # Calc number of batches
    num_batches = int(len(dataset) / data_loader.batch_size)

    # Init tqdm
    tk0 = tqdm(data_loader, total=num_batches)

    # Loop over all batches
    for i, d in enumerate(tk0):
        inputs = d["image"]
        targets = d["mask"]

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        try:
            targets = torch.reshape(targets, (data_loader.batch_size, 544, 960))
        except:
            print(f"targets size: {targets.size()}")
            print(f"batch size: {data_loader.batch_size}")
            print(f"inputs size: {inputs.size()}")
            print(f"outputs size: {outputs.size()}")
            print(f"index: {i}")

        loss = criterion(outputs, targets)
        loss.backward()


        optimizer.step()
    
    tk0.close()

def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            output = model(inputs)

            targets = torch.reshape(targets, (data_loader.batch_size, 544, 960))
            loss = criterion(output, targets)
            final_loss += loss
    tk0.close()

    return final_loss / num_batches

if __name__ == "__main__":
    df_classes = pd.read_csv(CLASSES_CSV, header=None)
    class_list = list(df_classes.iloc[:,0].values)

    df = pd.read_csv(TRAINING_CSV, header=None)
    df_train, df_val = model_selection.train_test_split(df, random_state=42, test_size=0.1)

    training_images = df_train.iloc[:, 0].values
    validation_images = df_val.iloc[:, 0].values

    model = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = len(class_list),
        activation = "sigmoid",
    )

    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    model.to(DEVICE)
    train_dataset = SIIMDataset(
        training_images,
        class_list,
        preprocessing_fn=prep_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )

    val_dataset = SIIMDataset(
        validation_images,
        class_list,
        transform=False,
        preprocessing_fn=prep_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )

    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1e-3
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )

    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Num training images: {len(train_dataset)}")
    print(f"Num validation images: {len(val_dataset)}")
    print(f"Encoder: {ENCODER}")
    print(f"Encoder weights: {ENCODER_WEIGHTS}")

    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch}")
        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer
        )
        print(f"Validation Epoch: {epoch}")
        val_log = evaluate(
            val_dataset, val_loader, model
        )
        scheduler.step(val_log)
        print("\n")