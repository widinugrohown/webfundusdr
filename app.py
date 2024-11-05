from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from model import model
import os
import torch.nn as nn

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = model()
model.load_state_dict(torch.load(r'D:\projects\webfundusdr\model\model_dr.pth',torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = ['Mild', 'Moderate', 'No DR', 'PDR', 'Severe']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
            prediction_probability = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

        return jsonify({
            'class_name': predicted_class,
            'prediction_probability': prediction_probability,
            'image_path': url_for('static', filename=f'uploads/{filename}')
        })

if __name__ == '__main__':
    app.run(debug=True)
