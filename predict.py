# maptplotlib
import matplotlib.pyplot as plt
#data structures and utils
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
#torch
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
#parameters Loading
from AppParametersLoader import AppParametersLoader
parameters = AppParametersLoader()
parameters.print_all()
#Data Loading
from DataManager import DataManager
data_manager = DataManager()
#previous saved model loading
from ModelManager import ModelManager
chk_path = parameters.save_dir() + '/checkpoint.pth'
model_manager = ModelManager(parameters.gpu())
model_manager.load_model(chk_path)
#prediction
image_to_predict = data_manager.process_image(parameters.image_path())
top_ps, top_classes = model_manager.predict(image_to_predict, topk=parameters.top_k())
#Getting categories names by class labels
cat_to_name = data_manager.get_cat_to_name(parameters.category_names_path())
category_names = []
for i in top_classes:
    category_names += [cat_to_name[i]]
print(f"Predicition: probabilities: {top_ps} ")
print(f"Prediction: classes: {category_names} ")