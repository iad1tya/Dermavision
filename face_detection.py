import cv2
import numpy as np

def detect_face(image_path):
    # Load the pre-trained Haar cascade model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # If a face is detected, crop the first detected face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]  # Crop the face
        return face
    else:
        return None  # No face detected

# Example usage:
if __name__ == "__main__":
    face = detect_face("test.jpg")
    if face is not None:
        cv2.imwrite("cropped_face.jpg", face)  # Save cropped face
        print("Face detected and saved!")
    else:
        print("No face detected.")
