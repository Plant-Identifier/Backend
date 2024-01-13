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

# number of epochs (iterations over training set)
num_epochs = 15  # You can adjust this

# for each epoch
for epoch in range(num_epochs):
    
    # set to training mode
    model.train()

    # set the loss
    running_loss = 0.0

    # loops over training batches
    # images contains input, labels contains what each image is
    for images, labels in train_loader:

        # sets gradient to zero
        optimizer.zero_grad()

        # performs forward pass, generates predictions for input images
        outputs = model(images)

        # computes loss, measures difference from model prediction to truth label
        loss = loss_function(outputs, labels)

        # backpropagation, gradients are computer for all model parameters
        loss.backward()

        # updates model parameter w/ optimization algorithm
        optimizer.step()

        # accumulates loss
        running_loss += loss.item()

    # print statistics
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # validate after each epoch
    # set to evaluation mode
    model.eval()

    # tracks correct & total images
    correct = 0
    total = 0

    # context manager disables gradient tracking
    with torch.no_grad():

        # loops over validation data
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on validation set: {100 * correct / total}%')

    torch.save(model.state_dict(), 'plantscout.pth')
    print("Model successfully saved to plantscout.pth")