from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
model = load_model('four_class_model_resnet.h5')  
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preduct')
def preduct():
    return render_template('preduct.html')
    
class_names = ['Corona Virus Disease', 'Normal', 'Pneumonia', 'Tuberculosis']

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        img_bytes = img.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224), color_mode='rgb')  
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        predictions = model.predict(x)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(predictions)

        return redirect(url_for('result2', predicted_class=predicted_class, confidence=confidence))

@app.route('/result2')
def result2():
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')
    return render_template('result2.html', predicted_class=predicted_class, confidence=confidence)

@app.route('/precautions2')
def precautions2():
    predicted_class = request.args.get('predicted_class')
    # Here you can define the precautions for each disease
    precautions_dict = {
        'Corona Virus Disease': [
            "Wash your hands frequently.",
            "Avoid touching your face, especially the eyes, nose, and mouth.",
            "Practice social distancing.",
            "Wear a mask in public.",
            "Cover your mouth and nose with your elbow or tissue when you cough or sneeze."
        ],
        'Normal': [
            "Maintain a healthy lifestyle.",
            "Exercise regularly.",
            "Eat a balanced diet.",
            "Get enough sleep.",
            "Avoid smoking and excessive alcohol consumption."
        ],
        'Pneumonia': [
            "Get vaccinated.",
            "Practice good hygiene to avoid infections.",
            "Don't smoke.",
            "Keep your immune system strong.",
            "Avoid close contact with people who have a cold, the flu, or other respiratory tract infections."
        ],
        'Tuberculosis': [
            "Finish your entire course of medication.",
            "Take precautions to protect your family and friends.",
            "Stay home and avoid other people as much as possible.",
            "Wear a mask.",
            "Ventilate the room."
        ]
    }
    precautions2 = precautions_dict.get(predicted_class)
    return render_template('precautions2.html', predicted_class=predicted_class, precautions=precautions2)

if __name__ == '__main__':
    app.run(debug=True, port=3000)


