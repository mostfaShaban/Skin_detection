from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the skin disease classification model during initialization
model = load_model(r"E:\FCAI\Graduation project\TRY\skin_cancer_detection7.h5")

# Define the classes for skin diseases
classes = {
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'Basal cell carcinoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('mel', 'Melanoma'),
    5: ('nv', 'Melanocytic nevi'),
    6: ('vasc', 'Pyogenic granulomas and hemorrhage')
}

# Function to preprocess the image for skin disease classification
def preprocess_image(image):
    image = image.resize((90, 120))  # Resize the image to match the model's input shape
    image_array = np.array(image)  # Convert image to array
    image_array = image_array / 255.0  # Normalize the pixel values
    return image_array

@app.route('/')
def index():
    return render_template('index.html', appName="Skin Disease Detection")

@app.route('/predict', methods=['POST'])
def predict():
    if 'fileup' not in request.files:
        return jsonify({'error': 'Please upload an image'})
    
    file = request.files['fileup']
    if file.filename == '':
        return jsonify({'error': 'Please upload a valid image'})
    
    try:
        image = Image.open(file)
        image_array = preprocess_image(image)
        # Make predictions
        predictions = model.predict(np.array([image_array]))
        # Interpret the predictions
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = classes[predicted_class_index][0]
        predicted_class_description = classes[predicted_class_index][1]
        return jsonify({'prediction': predicted_class_name, 'description': predicted_class_description})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
