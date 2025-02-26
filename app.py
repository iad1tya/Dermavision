from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from models.skin_analysis import analyze_skin
from database import database

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze skin from uploaded image"""
    data = request.json
    image_data = data['image']
    user_name = data['name']
    user_age = data['age']

    # Decode base64 image
    image_decoded = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(image_decoded, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Analyze skin condition
    skin_condition = analyze_skin(img)

    # Save user data to MongoDB
    database.save_user_data(user_name, user_age, skin_condition)

    return jsonify({"name": user_name, "age": user_age, "skin_condition": skin_condition})

if __name__ == "__main__":
    app.run(debug=True)