from options import options_train
import numpy as np
import pandas as pd
import matplotlib
# import getattr(matplotlib, 'x') as plt

import json
import os
import torch
import importlib

from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
print(transforms.__getattr__('Compose'))

print("Finished importing python packages")
args = options_train()
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"
def main():
    model()

def prepare_data():
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
    image_datasets = {"train_data": datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                      "valid_data": datasets.ImageFolder(valid_dir, transform=data_transforms['valid_test_transforms']),
                      "test_data": datasets.ImageFolder(test_dir, transform=data_transforms['valid_test_transforms'])}
    # define dataloaders
    dataloaders = {
        "train_loader": torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        "valid_loader": torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64, shuffle=True),
        "test_loader": torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64, shuffle=True)}
    print("Data loading finished")
    return dataloaders

def build_model():
    model_name = args.arch
    try:
        model_module = importlib.import_module(f"torchvision.models.{model_name}")
        model = getattr(model_module, model_name)()
    except (ImportError, AttributeError) as e:
        print(f"Error importing model: {e}")
    print("Finished loading model")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # create a classifier
    hidden_units = args.hidden_units
    classifier = nn.Sequential(OrderedDict([
        ("layer1", nn.Linear(1664, hidden_units)),
        ("relu", nn.ReLU()),
        ("dropout", nn.Dropout(p = 0.2)),
        ("layer2", nn.Linear(hidden_units, 102)),
        ("output", nn.LogSoftmax(dim = 1))
    ]))
    model.classifier = classifier
    return model

def train_model():
    model = build_model()
    dataloaders = prepare_data()
    # choose either gpu or cpu
    gpu = args.gpu
    if gpu == 'True':
        if torch.cuda.is_available():
            processor = 'cuda'
        else:
            print("cuda is not available, defaulted to cpu")
            processor = 'cpu'

    else:
        processor = 'cpu'
    # Define the loss metrics and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    model.to(processor)

    # Training the model
    epochs = args.epochs
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for e in range(epochs):
        train_loss = 0
        valid_loss = 0
        valid_accuracy = 0

        for images, labels in dataloaders['train_loaders']:
            images, labels = images.to(processor), labels.to(processor)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        else:
            model.eval()
            for images, labels in dataloaders['valid_loaders']:
                images, labels = images.to(processor), labels.to(processor)
                with torch.no_grad():
                    logps = model.forward(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_k = ps.topk(1, dim = 1)
                    equals = top_k == labels.view(*top_k)
                    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                    valid_accuracy += accuracy

            train_loss /= len(dataloaders['train_loader'])
            valid_loss /= len(dataloaders['valid_loader'])
            valid_accuracy /= len(dataloaders['valid_loader'])

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print("Epoch: {}".format(e),
                  "Train loss: {}".format(train_loss),
                  "Test Loss: {}".format(valid_loss),
                  "Test_accuracy: {}".format(valid_accuracy))
            model.train()
    return model

def model():
    # Label mapping
    with open(project_path + "/cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)
    print("Loaded cat_to_name json file")

    # Load the model from build model function
    model = train_model()















if __name__ == "__main__":
    main()