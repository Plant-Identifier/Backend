#for reading csv
import pandas as pd 
#for pytorch
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import models
#for operating system
import os 
#for images 
from PIL import Image 
#optimization
import torch.optim as optim

#transform the images for pytorch
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

#get the dataset 
dataset = datasets.ImageFolder(root= 'path', transform=transform)

#create the dataloader
train_loader = DataLoader('path', batch_size=64, shuffle = True)

#call the pretrained image classification model
model = models.resnet18(pretrained=True)

#reset the final section of the model to match our dataset 
num_classes = 62
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

#creating the loss and optimization functions 
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)





