#for reading csv
import pandas as pd 
#for pytorch
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
#for operating system
import os 
#for images 
from PIL import Image 
#optimization
import torch.optim as optim

#transform the images for pytorch
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

#get the dataset 
dataset = datasets.ImageFolder(root= 'plants\datasets\dataset', transform=transform)
validataset = datasets.ImageFolder(root= 'plants\datasets\dataset-test', transform=transform)

#create the dataloader
train_loader = DataLoader(dataset, batch_size=64, shuffle = True)
val_loader = DataLoader(validataset, batch_size = 64)

#call the pretrained image classification model
model = models.resnet18(pretrained=True)

#reset the final section of the model to match our dataset 
num_classes = 62
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

#creating the loss and optimization functions 
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)


num_epochs = 10  # You can adjust this

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = loss_function(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        running_loss += loss.item()

    # Print statistics
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # Validate after each epoch
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient tracking
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on validation set: {100 * correct / total}%')



