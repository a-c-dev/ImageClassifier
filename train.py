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
data_manager.load_TrainTestValid(parameters.data_dir())

#model definition
from ModelManager import ModelManager
if parameters.arch() == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_nodes = 25088
elif parameters.arch() == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_nodes = 1024
    
classifier = nn.Sequential(nn.Linear(input_nodes, parameters.hidden_units()),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(parameters.hidden_units(), len(data_manager.get_images_data_training().class_to_idx)),
                           nn.LogSoftmax(dim=1)
                           )
model_manager = ModelManager(parameters.gpu())
model_manager.set_model(model, classifier, parameters.learning_rate())

#model training
model_manager.train(parameters.epochs(),
                    {'training': data_manager.get_dataloader_training(),
                     'validation': data_manager.get_dataloader_validation()
                    }
                   )
#model testing
accuracy = model_manager.test_accuracy(data_manager.get_dataloader_testing())

if accuracy >= 0.7:
    print("Accuracy over 70 perc. Good model")
else:
    print("Accuracy over 70 perc. Bad model")

chk_path = parameters.save_dir() + '/checkpoint.pth'
model_manager.save_checkpoint(chk_path,
                              data_manager.get_images_data_training().class_to_idx,
                              parameters.arch()
                              )
print(f"Model saved in checkpoint {chk_path}")



