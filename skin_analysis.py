from face_recognition import recognize_user, register_user
import cv2
import pymongo
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["skin_analysis_db"]
users_collection = db["users"]

def analyze_skin(image_path):
    """Analyze skin condition & identify user automatically."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Invalid image path or file."}

        # Recognize the user
        user, message = recognize_user(image)

        if user is None:
            return {"error": message}  # Either "No face detected" or "User not found"

        user_name = user["user_name"]

        # Perform skin analysis (reusing previous code)
        skin_condition = "Healthy Skin"  # Replace with CNN model inference

        # Save result in MongoDB
        result_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "skin_condition": skin_condition
        }
        
        try:
            users_collection.update_one(
                {"user_name": user_name}, 
                {"$push": {"past_results": result_data}}
            )
        except pymongo.errors.PyMongoError as e:
            return {"error": f"Database error: {str(e)}"}
        
        return {
            "user_name": user_name, 
            "skin_condition": skin_condition, 
            "message": "Analysis saved."
        }
        
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
