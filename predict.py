import face_recognition
import pickle
import os
import numpy as np
import cv2
import threading
import model as face_model

MODEL_DIR = "face_models"
MATCH_THRESHOLD = 0.4

def load_models():
    """
    Load all saved face recognition models.
    Returns:
        dict: Dictionary mapping person names to their face encodings
    """
    models = {}
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith("_model.pkl"):
            person_name = model_file.replace("_model.pkl", "")
            with open(os.path.join(MODEL_DIR, model_file), 'rb') as f:
                models[person_name] = pickle.load(f)
    return models

def enhance_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.5, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def predict_faces(image_path, models):
    if not os.path.exists(image_path):
        return [], None
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model='hog')
    face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=10)
    if not face_locations:
        image = enhance_brightness(image)
        face_locations = face_recognition.face_locations(image, model='hog')
        face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=10)
    identified_persons = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        person_name = "unknown"
        min_distance = float('inf')
        accuracy = 0.0
        for name, person_encodings in models.items():
            distances = face_recognition.face_distance(person_encodings, face_encoding)
            current_min = np.min(distances)
            if current_min < min_distance and current_min <= MATCH_THRESHOLD:
                min_distance = current_min
                person_name = name
                accuracy = max(0.0, (1.0 - current_min) * 100)
        identified_persons.append((person_name, accuracy, face_location))
    return identified_persons, image

def draw_results(image, identified_persons):
    """Draw bounding boxes and labels on the image. Returns a BGR image for web display."""
    image = image.copy()
    for person_name, accuracy, (top, right, bottom, left) in identified_persons:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        if person_name == "unknown":
            label = "unknown"
        else:
            label = f"{person_name} ({accuracy:.2f}%)"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (left, top - text_height - 10), (left + text_width, top), (0, 255, 0), -1)
        cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # Convert RGB to BGR for OpenCV imencode
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def background_train(image_paths, person_name):
    face_model.train_and_save_model(image_paths, person_name)

# CLI and Tkinter code removed for web use
if __name__ == "__main__":
    print("This module is intended to be imported for web usage. CLI is disabled.")