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
    predict()

def load_cat_to_name():
    # Label mapping
    with open(project_path + "/" + args.category_names, "r") as f:
        cat_to_name = json.load(f)
    print("Loaded cat_to_name json file")
    return cat_to_name

def load_class_to_idx():
    with open(project_path + "/class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)
    return class_to_idx

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
    print("Opened image")

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
            processor = 'cuda'
            print("gpu available")
        else:
            processor = 'cpu'
            print('gpu unavailable, defaulted to cpu')
    else:
        processor = 'cpu'
    image = preprocess_image()
    model = load_model()
    class_to_idx = load_class_to_idx()
    cat_to_name = load_cat_to_name()
    top_k = args.top_k
    
    image.to(processor)
    
    image = image.unsqueeze(0)
    logps = model.forward(image)

    ps = torch.exp(logps)
    top_p, top_k = ps.topk(top_k, dim = 1)


    top_predicted = [list(class_to_idx.keys())[i] for i in top_k.view(-1)]
    top_predicted = [cat_to_name[i] for i in top_predicted]
    print("TOP PREDICTION")
    for i, j in zip(top_predicted, top_p):
        print(f"Flower name: {top_predicted}, Probability: {j}")


if "__main__" == __name__:
    main()
