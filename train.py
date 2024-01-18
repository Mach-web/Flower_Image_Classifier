from options import options_train
from build_models import build_model
from prepare_data import prepare_data

import matplotlib.pyplot as plt

import os
import torch

from torch import nn, optim

print("Finished importing python packages")

# Define global variables
args = options_train()
project_path = os.getcwd()
model_name = args.arch
data_dir = args.data_dir
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"
def main():
    save_model()


def train_model():
    dataloaders = prepare_data(project_path, data_dir)
    print("Data loading finished")

    model = build_model(model_name, args.hidden_units)
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
   
        for images, labels in dataloaders['train_loader']:
            images, labels = images.to(processor), labels.to(processor)
            optimizer.zero_grad()
         
#             logps = model(images)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            train_loss += loss.item()
        
            loss.backward()
            optimizer.step()

        else:
            model.eval()
            for images, labels in dataloaders['valid_loader']:
                images, labels = images.to(processor), labels.to(processor)
                with torch.no_grad():
                    logps = model.forward(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_k = ps.topk(1, dim = 1)
                    equals = top_k.view(-1) == labels
                    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                    valid_accuracy += accuracy

            train_loss /= len(dataloaders['train_loader'])
            valid_loss /= len(dataloaders['valid_loader'])
            valid_accuracy /= len(dataloaders['valid_loader'])

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print("Epoch: {}".format(e+1),
                  "Train loss: {}".format(train_loss),
                  "Test Loss: {}".format(valid_loss),
                  "Test_accuracy: {}".format(valid_accuracy))
            model.train()
    return model, processor

def test_model():
    dataloaders = prepare_data(project_path, data_dir)
    model, processor = train_model()
    model.eval()
    test_accuracy = 0
    for images, labels in dataloaders['test_loader']:
        images, labels = images.to(processor), labels.to(processor)
        with torch.no_grad():
            logps = model.forward(images)

            ps = torch.exp(logps)
            top_p, top_k = ps.topk(1, dim=1)
            equals = top_k.view(-1) == labels
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))
    print(f"The test accuracy is: {test_accuracy/len(dataloaders['test_loader'])}")
    return model

def save_model():
    model = test_model()
    torch.save(model.state_dict(), project_path + args.save_dir + args.modelName)
    print(f"Model saved successfully as {args.modelName}")

if __name__ == "__main__":
    main()
