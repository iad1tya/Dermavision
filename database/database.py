from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["skin_analysis_db"]
collection = db["user_data"]

def save_user_data(name, age, skin_condition):
    """Save user data to MongoDB"""
    user_entry = {
        "name": name,
        "age": age,
        "skin_condition": skin_condition,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(user_entry)
    print(f"✅ Data saved for {name}")
