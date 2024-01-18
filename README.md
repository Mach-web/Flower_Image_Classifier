# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Data

This project uses flower data which can be ([downloaded here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz))

For this project, I have named the data folder  '/flower_data'.


# Specifications

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. 


The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.


# Files

### Build_models.py

Contains three models that can be built and trained i.e densenet201, densenet169, and vgg16. Densenet201 had the highest accuracy of about 0.92 while vgg16 had the lowest.

### class_to_idx.json

This a file create after preparing the data. Its a mapping of classes to indices which you get from one of the image datasets: image_datasets['train'].class_to_idx. 

### Image_Classifier_Project.ipynb

Trained a neural network to identify flower species.

### Options.py

Gets the command line input into the scripts.

### Predict.py 

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

### Prepare_data.py

Load and preprocess the image dataset. Also loads the class_to_idx.json file.

### Train.py

Train a new network on a data set with train.py.

