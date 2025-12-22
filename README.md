# hand-sign-recognition-and-classification

This project is a simple web-based application that recognizes hand signs using machine learning and computer vision. It uses a trained TensorFlow model to classify hand gestures captured through a webcam and displays the predicted sign in real time using a Flask web app.

The goal of this project is to explore hand gesture recognition and understand how machine learning models can be integrated into web applications.

What This Project Does

Detects hand signs using a webcam
Classifies gestures using a trained TensorFlow model
Shows predictions through a web interface
Demonstrates basic integration of ML models with Flask
Technologies Used
Python
Flask – for the web application
TensorFlow – for hand sign classification
HTML, CSS, JavaScript – for the front end
OpenCV – for handling image/video input

Project Structure
hand-sign-recognition-and-classification/

├── app.py                # Flask application


├── class_model.py        # Model loading and prediction logic

├── model/                # Trained ML model files

├── static/

│   └── css/              # Stylesheets

├── templates/            # HTML files

└── README.md             # Project documentation


How It Works

The webcam captures live video frames
The frames are processed and passed to the trained model
The model predicts the hand sign
The result is displayed on the web page in real time

Use Cases

Learning computer vision and hand gesture recognition
Beginner-friendly ML + Flask integration example
Foundation for sign language recognition projects

