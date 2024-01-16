import importlib
from collections import OrderedDict
from torch import nn
from torchvision import models

def build_model(model_name):
    try:
        model_module = importlib.import_module(f"torchvision.models.{model_name}")
        model = getattr(model_module, model_name)()
        print("Finished loading model")
    except (ImportError, AttributeError) as e:
        print(f"Error importing model: {e}")

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