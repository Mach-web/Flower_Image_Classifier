import importlib
from collections import OrderedDict
from torch import nn
from torchvision import models
from options import options_train

def build_model(model_name, hidden_units = 512):
    try:
        model = getattr(models, model_name)(pretrained=True)
        print("Finished loading model")
    except (ImportError, AttributeError) as e:
        print(f"Error importing model: {e}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

#     return build_densenet169(model, hidden_units)
#     return build_vgg16(model, hidden_units)
    return build_densenet201(model, hidden_units)
    
def build_densenet169(model, hidden_units):
    # create a classifier
    classifier = nn.Sequential(OrderedDict([
        ("layer1", nn.Linear(1664, hidden_units)),
        ("relu", nn.ReLU()),
        ("dropout", nn.Dropout(p = 0.2)),
        ("layer2", nn.Linear(hidden_units, 102)),
        ("output", nn.LogSoftmax(dim = 1))
    ]))
    model.classifier = classifier
    return model

def build_vgg16(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('layer1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p = 0.2)),
        ('layer2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('layer3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    model.classifier = classifier
    return model

def build_densenet201(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('layer1', nn.Linear(1920, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p = 0.2)),
        ('layer3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    model.classifier = classifier
    return model

