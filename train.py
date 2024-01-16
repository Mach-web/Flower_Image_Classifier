from options import options_train
from build_models import build_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import torch
import importlib

from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

print("Finished importing python packages")

# Define global variables
args = options_train()
project_path = os.getcwd()
model_name = args.arch
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"
def main():
    images = prepare_data()
    print(images['train_data'].class_to_idx)
    # save_model()

def prepare_data():
    # Define data paths
    data_dir = project_path + "/" + args.data_dir
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
    return dataloaders


def train_model():
    dataloaders = prepare_data()
    print("Data loading finished")

    model = build_model(model_name)
    print("Model built successfully")
    # choose either gpu or cpu
    gpu = args.gpu
    if gpu == 'True':
        if torch.cuda.is_available():
            processor = 'cuda'
            print('cuda available')
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
    plt.plot(train_losses, label='Training loss');
    plt.plot(valid_losses, label='Validation loss');
    plt.legend(frameon=False);
    return model

def test_model():
    dataloaders = prepare_data()
    model = train_model()
    model.eval()
    test_accuracy = 0
    for images, labels in dataloaders['test_loader']:
        with torch.no_grad():
            logps = model.forward(images)

            ps = torch.exp(logps)
            top_p, top_k = ps.topk(1, dim=1)
            equals = top_k.flatten() == labels
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))
    print(f"The test accuracy is: {test_accuracy/len(dataloaders['test_loaders'])}")
    return model

def save_model():
    model = test_model()
    torch.save(model.state_dict(), project_path + args.save_dir + args.modelName)
    print(f"Model saved successfully as {args.modelName}")

if __name__ == "__main__":
    main()