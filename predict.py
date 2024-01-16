from options import options_predict
from build_models import build_model

from PIL import Image
from torchvision import transforms

import os
import json
import torch

# define global variables
args = options_predict()
project_path = os.getcwd()
def main():
    # unpack values in parser
    print(args)

def load_json():
    # Label mapping
    with open(project_path + "/" + args.category_names, "r") as f:
        cat_to_name = json.load(f)
    print("Loaded cat_to_name json file")
    return cat_to_name

def load_model():
    checkpoint_name = args.checkpoint
    state_dict = torch.load(project_path + "/saved_models/" + checkpoint_name)
    model = build_model(args.arch)
    try:
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except:
        print("Keys did not match")
    return model

def preprocess_image():
    image = Image.open(args.path_image)

    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_image(image)

def predict():
    gpu = args.gpu
    if gpu == 'True':
        if torch.cuda.is_available():
            processor = 'gpu'
            print("gpu available")
        else:
            processor = 'cpu'
            print('gpu unavailable, defaulted to cpu')
    else:
        processor = 'cpu'
    image = preprocess_image()
    model = load_model()
    top_k = args.top_k

    logps = model.forward(image)

    ps = torch.exp(logps)
    top_p, top_k = ps.topk(top_k, dim = 1)




if "__main__" == __name__:
    main()
