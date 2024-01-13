import json
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
# for operating system
import os 
# for images 
from PIL import Image 

transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

# Load the trained model
model = models.resnet18(pretrained=False)  # Initialize the model
model.fc = torch.nn.Linear(model.fc.in_features, 62)  # Adjust final layer
model.load_state_dict(torch.load('plantscout.pth'))  # Load trained weights
model.eval()  # Set to evaluation mode

def predict_image(image_path, model, class_names, transform):
    image = Image.open(image_path)  # Load the image
    image = transform(image).unsqueeze(0)  # Apply transformation and add batch dimension
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(image)  # Forward pass
        _, predicted_class_index = torch.max(outputs, 1)  # Get the index of the max log-probability

    predicted_class = class_names[predicted_class_index.item()]  # Retrieve the class label
    return predicted_class

# Load the class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Load the trained model
model.load_state_dict(torch.load('plantscout.pth'))

# User interaction to select an image and get a prediction
predicted_class = predict_image('plants\datasets\dataset-user_images\lambo.jpg', model, class_names, transform)  # Make a prediction

print(f'The predicted class for the image is: {predicted_class}')  # Output the result
