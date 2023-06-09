


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
num_classes = 19
datasetclasses=['Cereus nashii','Cleistocactus icosagonus','Copiapoa taltalensis','Copiapoa tocopillana','Cylindropuntia chuckwallensis','Disocactus nelsonii','Epiphyllum phyllanthus var. hookeri','Eriosyce curvispina var. tuberisulcata','Harrisia gracilis','Leuchtenbergia principis','Miqueliopuntia miquelii','Opuntia azurea var. diplopurpurea','Opuntia soederstromiana','Pereskia humboldtii var. humboldtii','Rhipsalis baccifera subsp. baccifera','Rhipsalis oblonga','Stenocereus yunckeri','Tephrocactus aoracanthus','Weberocereus rosei']

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = models.mobilenet_v2(pretrained=False)
best_model.classifier[1] = nn.Linear(best_model.classifier[1].in_features, num_classes)

# Load the model weights
model_path = '/Users/adityakanodia/Desktop/DatVis/newproject/best_model.pth'
best_model.load_state_dict(torch.load(model_path, map_location=device))

# Move the model to the appropriate device
best_model = best_model.to(device)
best_model.eval()

# Define the transformation for preprocessing the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = 'image2.jpeg'
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

# Perform the prediction
with torch.no_grad():
    output = best_model(image)
    _, predicted = torch.max(output.data, 1)

print(predicted.item())
# Get the predicted class label
predicted_label = datasetclasses[predicted.item()]

print('Predicted Label:', predicted_label)


'''
from PIL import Image

from torchvision import models
import torch.nn as nn
import torch
import torchvision
num_classes = 19
best_model = models.mobilenet_v2(pretrained=False)
best_model.classifier[1] = nn.Linear(best_model.classifier[1].in_features, num_classes)
best_model.load_state_dict(torch.load('/Users/adityakanodia/Desktop/DatVis/newproject/best_model.pth'))
best_model = best_model.to(device)
best_model.eval()

# Load and preprocess the image
image_path = 'image1.jpeg'
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

# Perform the prediction
with torch.no_grad():
    output = best_model(image)
    _, predicted = torch.max(output.data, 1)

# Get the predicted class label
predicted_label = dataset.classes[predicted.item()]

print('Predicted Label:', predicted_label)

'''