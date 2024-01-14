import json
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn.functional as F
# for operating system
import os 
# for images 
from PIL import Image 
from flask import Flask, jsonify
from flask import request
from werkzeug.utils import secure_filename

# list of instructions for how to prepare pictures
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

# load the trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 62)
model.load_state_dict(torch.load('plantscout.pth'))

# sets model to guessing mode
model.eval()

# defines a function to predict the image that takes a path, uses the pretrained model, and applies a transform to each image
def predict_image(image_path, model, class_names, transform):
    # opens the image based on the path
    image = Image.open(image_path)

    # applies the transform to the image
    image = transform(image).unsqueeze(0)
    
    # by setting torch to no gradient, no need to remember the process to its guesses
    with torch.no_grad():
        # forward pass -> makes a guess
        outputs = model(image)

        # convert logits into percentage
        probabilities = F.softmax(outputs, dim=1)

        # set top_prob & top_class_idx to whichever percentage is the highest (value & index, which will always be 1)
        top_prob, top_class_idx = torch.max(probabilities, 1)
        
        # match the predicted class to the index's guess
        predicted_class = class_names[top_class_idx.item()]
        
        # determine confidence
        confidence = top_prob.item()

    # conditional output based on confidence
    if confidence < 0.5:
        print("unknown, please retry.")
    
    else:
        print(f'The predicted class for the image is: {predicted_class} with confidence of {confidence:.2%}')
        return predicted_class

# load the class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# load the trained model
model.load_state_dict(torch.load('plantscout.pth'))

app = Flask(__name__)

#NOT CURRENTLY WORKING NEED TO DO MORE TESTING THIS IS A DRAFT 
@app.route('/evaluate', methods=['POST'])
def evaluate():
    return("test")

    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    print(file)

    if file.filename == '':
        return "No selected file", 400
    
    filep = os.path.join('/tmp', filename)
    print(filep)
    file.save(filep)

    # predict image based on file path (current set statically, change for dynamic)
    predicted_class = predict_image(filepath, model, class_names, transform)

    os.remove(filep)

    return jsonify(predicted_class)
