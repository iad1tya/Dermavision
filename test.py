import cv2
import face_recognition
from face_utils import register_user

image = cv2.imread("test.jpg")
if image is None:
    raise FileNotFoundError("Could not load image file 'test.jpg'")

register_user(image, "Aditya Yadav")
