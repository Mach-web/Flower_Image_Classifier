import argparse


def options_train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="/flower_data")
    parser.add_argument("--arch", type=str, default="densenet169")
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--hidden_units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpu", default='True')
    return parser.parse_args()

def options_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--category_names", type=str, default="cat_to_name.json")
    parser.add_argument("--gpu", default='True')
    # return namespace of parser
    return parser.parse_args()
