from flask import Flask, request, render_template, redirect, url_for, session
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd

app = Flask(__name__, static_url_path='/static')

num_classes = 19
datasetclasses=['Cereus nashii','Cleistocactus icosagonus','Copiapoa taltalensis','Copiapoa tocopillana','Cylindropuntia chuckwallensis','Disocactus nelsonii','Epiphyllum phyllanthus var. hookeri','Eriosyce curvispina var. tuberisulcata','Harrisia gracilis','Leuchtenbergia principis','Miqueliopuntia miquelii','Opuntia azurea var. diplopurpurea','Opuntia soederstromiana','Pereskia humboldtii var. humboldtii','Rhipsalis baccifera subsp. baccifera','Rhipsalis oblonga','Stenocereus yunckeri','Tephrocactus aoracanthus','Weberocereus rosei']

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = models.mobilenet_v2(pretrained=False)
best_model.classifier[1] = nn.Linear(best_model.classifier[1].in_features, num_classes)

# Load the model weights
model_path = 'best_model.pth'
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


app = Flask(__name__)



@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/successname/<name>')
def successname(name):
    if name:
        df=pd.read_csv('dataset.csv')
        df=df[['Species','lat','lon','Description','imag','TRUE']]
        df.to_json('static.json',orient='records')
        jsondata=df[df['Species']==name].to_json(orient='records')
        print(jsondata)
    return render_template("result.html",data=jsondata)
        


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['speciesimage']
    species_name = request.form["speciesname"]

    if image_file.filename:
        try:
            image = Image.open(image_file)
            image = transform(image).unsqueeze(0).to(device)

            # Perform the prediction
            with torch.no_grad():
                output = best_model(image)
                _, predicted = torch.max(output.data, 1)
            predicted_label = datasetclasses[predicted.item()]
            print('Predicted Label:', predicted_label)
            return redirect(url_for('successname', name=predicted_label))
        except (IOError, SyntaxError) as e:
            print("Invalid Image:", str(e))
            return render_template("index.html")

    elif species_name:
        return redirect(url_for('successname', name=species_name))

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)


'''
from flask import Flask, request, render_template,redirect, url_for
from PIL import Image
app = Flask(__name__)

@app.route('/')
def hello_world():
   return render_template("index.html")

@app.route('/success/<data>')
def success(data):
   return 'welcome %s' % data

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    if request.method == 'POST':
        image_file = request.files['speciesimage']
        species_name = request.form["speciesname"]


        if not image_file.filename and species_name=='':
            print("No Date")
            return render_template("index.html")

        # Validate if the file is an image
        if image_file.filename:
            try:
                image = Image.open(image_file)
                print("Valid")
                image.verify() 
                redirect(url_for('success',data=image)) 
            except (IOError, SyntaxError) as e:
                print("Invalid Image:", str(e))

                return render_template("index.html")
        elif species_name!='':
            print(species_name)
            return render_template("index.html")

        

        return render_template("index.html")





if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)
'''