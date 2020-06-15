import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.models as models

class ModelManager:
    def __init__(self, gpu_usage):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_usage else "cpu")
    
    #init model manager by model parameters
    def set_model(self, architecture, classifier, learning_rate):
        self.model = architecture
        for param in self.model.parameters():
             param.requires_grad = False
        self.model.classifier = classifier
        self.model.to(self.device)
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        
    #init modelmanager by checkpoint file
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model = getattr(models, checkpoint['pre_trained_network'])(pretrained=True)
        self.model.classifier = checkpoint['classifier']
        self.model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.epochs = checkpoint['epochs']
        self.learning_rate = checkpoint['learning_rate']
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.to(self.device)
        self.criterion = nn.NLLLoss()
        
        
    def train(self, epochs, dataloaders):
     print(" START TRAINING... ")
     self.epochs = epochs
     print_every = 10
     training_loop = 0
     for epoch in range(self.epochs):#for each epoch
        running_loss = 0
        for images_train, labels_train in dataloaders['training']: #for each batch in training data loader
            images_train, labels_train = images_train.to(self.device), labels_train.to(self.device)
            #--------------------------------------------------- TRAINING ----------------
            #reset gradients on optimizer
            self.optimizer.zero_grad() 
            #feedforward
            log_ps = self.model.forward(images_train)
            #loss and gradients calculation
            loss = self.criterion(log_ps, labels_train)
            loss.backward()
            #updating weights
            self.optimizer.step()
            #adding batch loss to total loss
            running_loss += loss.item()
            training_loop +=1
            #--------------------------------------------------- VALIDATION ----------------
            if training_loop  % print_every == 0:
                #setting evaluation mode (dropout off)
                self.model.eval()
                valid_loss = 0
                accuracy = 0
                #disabling gradients (useless gradients here, code runs faster)
                with torch.no_grad():
                    for images_valid, labels_valid in dataloaders['validation']: #fetching validation data
                        images_valid, labels_valid = images_valid.to(self.device), labels_valid.to(self.device)
                        #Loss Calculation
                        log_ps = self.model.forward(images_valid)
                        batch_loss = self.criterion(log_ps, labels_valid)
                        valid_loss += batch_loss.item()
                        #Accuracy Calculation
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_valid.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #setting training mode again (end of evaluation)       
                self.model.train()
                #print statement
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['validation']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['validation']):.3f}")
                running_loss = 0
                #End Torch No_grad
            #End if (End validation)
        #print statement for the epoch
        print(f" END OF EPOCH {epoch+1}/{epochs}.. ")
        #End epoch
     print(" END TRAINING... ")
     #End Training
    
    def test_accuracy(self, testloader):
        #setting evaluation mode (dropout off)
        self.model.eval()
        test_loss = 0
        accuracy = 0
        #disabling gradients (useless gradients here, code runs faster)
        with torch.no_grad():
            for images_test, labels_test in testloader: #fetching test-data
                images_test, labels_test = images_test.to(self.device), labels_test.to(self.device)
                #Loss Calculation
                log_ps = self.model.forward(images_test)
                batch_loss = self.criterion(log_ps, labels_test)
                test_loss += batch_loss.item()
                #Accuracy Calculation
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels_test.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #setting training mode again (end of evaluation)       
        self.model.train()
        #print statement
        print(f"Test accuracy: {accuracy/len(testloader):.3f}")
        return accuracy/len(testloader)
     
    def save_checkpoint(self, path, class_to_idx, arch_name):
        checkpoint = {
                        'pre_trained_network': arch_name, # self.model.__class__.__name__ returns VGG not vgg16 for example
                        'model_state_dict': self.model.state_dict(),
                        'classifier': self.model.classifier,
                        'classifier_state_dict': self.model.classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'class_to_idx': class_to_idx,
                        'learning_rate': self.learning_rate,
                        'epochs': self.epochs
                      }
        torch.save(checkpoint, path)
        
    def predict(self, image, topk=5):
        #loading and processing image (Transformation and tensor creation)
        if self.device == 'cpu':
            image_to_predict = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            image_to_predict= torch.from_numpy(image).type(torch.cuda.FloatTensor)
        image_to_predict = image_to_predict.unsqueeze_(0)
        self.model.to(self.device)
        image_to_predict.to(self.device)
        #settimg model evaluation mode (dropout off)
        self.model.eval()
        #disabling gradients: faster code
        with torch.no_grad():
            #feedforward in the network: fetching results (prediction)
            prediction = self.model.forward(image_to_predict)
            probabilities = torch.exp(prediction)
            top_ps= probabilities.topk(topk)[0].cpu().numpy()[0]
            top_classes_indexes = probabilities.topk(topk)[1].cpu().numpy()[0]
            #mapping class indexes to class labels
            idx_to_class= {x: y for y, x in self.model.class_to_idx.items()}
            top_classes_labels = []
            for clss in top_classes_indexes:
                top_classes_labels.append(idx_to_class[clss])
        #setting mode to the original one
        self.model.train()
        return top_ps, top_classes_labels