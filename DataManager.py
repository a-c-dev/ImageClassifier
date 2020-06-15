import torch
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json

class DataManager:
    def __init__(self):
        self.images_data = {}
        self.dataloaders = {}
        self.cat_to_name = {}
        
    def load_TrainTestValid(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = self.data_dir + '/train'
        self.valid_dir = self.data_dir + '/valid'
        self.test_dir = self.data_dir + '/test'

        images_transforms = {}
        images_transforms['training'] = transforms.Compose([transforms.RandomRotation(36),
                                                            transforms.RandomResizedCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                                 [0.229, 0.224, 0.225])])
        images_transforms['testing'] = transforms.Compose([transforms.Resize(264),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])])
        images_transforms['validation'] =  transforms.Compose([transforms.Resize(264),
                                                               transforms.CenterCrop(224),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                                    [0.229, 0.224, 0.225])])

        
        self.images_data['training'] = datasets.ImageFolder(self.train_dir, transform=images_transforms['training'])
        self.images_data['testing'] = datasets.ImageFolder(self.test_dir, transform=images_transforms['testing'])
        self.images_data['validation'] = datasets.ImageFolder(self.valid_dir, transform=images_transforms['validation'])

        self.dataloaders['training'] = torch.utils.data.DataLoader(self.images_data['training'], batch_size=62, shuffle=True)
        self.dataloaders['testing'] = torch.utils.data.DataLoader(self.images_data['testing'], batch_size=62)
        self.dataloaders['validation'] = torch.utils.data.DataLoader(self.images_data['validation'], batch_size=62)
   
    def get_images_data_training(self):
        return self.images_data['training']
    def get_images_data_validation(self):
        return self.images_data['validation']
    def get_images_data_testing(self):
        return self.images_data['testing']
    
    def get_dataloader_training(self):
        return self.dataloaders['training']
    def get_dataloader_validation(self):
        return self.dataloaders['validation']
    def get_dataloader_testing(self):
        return self.dataloaders['testing']
    
    def get_cat_to_name(self, cat_to_name_path):
        if len(self.cat_to_name) == 0:
            with open(cat_to_name_path, 'r') as f:
                self.cat_to_name = json.load(f)
        return self.cat_to_name
    
    def process_image(self, image_path):
        transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]
                                                            )
                                       ])
        return np.array(transform(Image.open(image_path)))
    
    def imshow(self, image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        return ax