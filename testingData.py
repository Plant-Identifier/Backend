import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models

# Transform the test images for pytorch
test_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

# Get the testing dataset (change the file path to the testing dataset file path)
test_dataset = datasets.ImageFolder(root='plants\datasets\dataset-testing', transform=test_transform)

# Create the data loader for the testing dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model - pretrained is set to False since we will import our own trained weights 
model = models.resnet18(pretrained=False) 
# Reset number of classes to match our dataset
num_classes = 62  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights from our model (change the file path to the file path of our model)
model.load_state_dict(torch.load('path-to-mode.pth'))
model.eval()

# Function to test the model 
def test(model, test_loader):
    # Counter for correct predictions
    correct = 0
    # Counter for total predictions
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test set: {100 * correct / total}%')

# Test the model 
test(model, test_loader)