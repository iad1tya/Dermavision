import face_recognition
import numpy as np
from database import users_collection
from sklearn.metrics.pairwise import cosine_similarity

def extract_face_embedding(image):
    """Extracts a 128-D face embedding vector using face_recognition"""
    rgb_image = image[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return None
    
    return face_encodings[0]

def register_user(image, user_name):
    """Register a new user with their face embedding"""
    embedding = extract_face_embedding(image)
    
    if embedding is None:
        return {"error": "No face detected"}

    users_collection.insert_one({
        "user_name": user_name,
        "face_embedding": embedding.tolist(),
        "past_results": []
    })
    return {"message": "User registered successfully"}

def recognize_user(image):
    """Finds the closest matching user in MongoDB"""
    new_embedding = extract_face_embedding(image)
    
    if new_embedding is None:
        return None, "Face not detected"

    users = users_collection.find()
    best_match = None
    best_similarity = 0.0

    for user in users:
        stored_embedding = np.array(user["face_embedding"])
        similarity = cosine_similarity([new_embedding], [stored_embedding])[0][0]

        if similarity > 0.6 and similarity > best_similarity:
            best_match = user["user_name"]
            best_similarity = similarity

    return best_match, None
