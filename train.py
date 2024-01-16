from options import options_train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import os
import torch
import collections

from torchvision import datasets, transforms, models
from torch import nn, optim

print("Finished importing python packages")
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"
def main():
    # unparse the parser
    options = options_train()
    model()
def model():
    args = options_train()

    # Define data paths
    project_path = os.getcwd()
    data_dir = project_path + args.save_dir
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
    # Transform the data
    data_transforms = {"train_transforms": transforms.Compose([transforms.RandomRotation(60),
                                                               transforms.RandomResizedCrop(224),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                                    [0.229, 0.224, 0.225])]),
                       "valid_test_transforms": transforms.Compose([transforms.Resize(255),
                                                                    transforms.CenterCrop(224),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                                         [0.229, 0.224, 0.225])])
                       }
    # load the dataset
    image_datasets = {"train_data": datasets.ImageFolder(train_dir, transform = data_transforms['train_transforms']),
                      "valid_data": datasets.ImageFolder(valid_dir, transform = data_transforms['valid_test_transforms']),
                      "test_data": datasets.ImageFolder(test_dir, transform = data_transforms['valid_test_transforms'])}
    # define dataloaders
    dataloaders = {"train_loader": torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                   "valid_loader": torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64, shuffle=True),
                   "test_loader": torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64, shuffle=True)}
    print("Data loading finished")
    # Label mapping
    with open(project_path + "/cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)
    print("Loaded cat_to_name json file")

    # Building and training a classifier
    # Load the model

if __name__ == "__main__":
    main()