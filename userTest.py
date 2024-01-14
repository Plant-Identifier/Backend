#needed import statements 
import json
import torch
from torchvision import transforms, models
import torch.nn.functional as F
#for images 
from PIL import Image

# List of instructions for how to prepare pictures
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

def load_model(model_path):
    # load the trained model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 62)
    model.load_state_dict(torch.load(model_path))
    # sets model to guessing mode
    model.eval()
    return model

# Defines a function to predict the image
def predict_image(image_path, model, class_names):
    # Opens the image based on the path
    image = Image.open(image_path)

    # Applies the transform to the image
    image = transform(image).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probabilities, 1)
        predicted_class = class_names[top_class_idx.item()]
        confidence = top_prob.item()

    #return the predicted class and confidence 
    return predicted_class, confidence

# Load the class names
def load_class_names(class_names_path):
    #get the class names from the file 
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    #return the class names 
    return class_names