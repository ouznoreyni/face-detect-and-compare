import face_recognition
import numpy as np
import os
from django.conf import settings

def load_known_face():
    """Load the known face from media/data/known_face.jpg"""
    known_face_path = os.path.join(settings.MEDIA_ROOT, 'data', '1.jpg')
    if os.path.exists(known_face_path):
        known_image = face_recognition.load_image_file(known_face_path)
        encodings = face_recognition.face_encodings(known_image)
        if encodings:
            return encodings[0]
    return None

def compare_with_known_face(uploaded_image_path):
    """Compare uploaded image with known face"""
    known_encoding = load_known_face()
    if known_encoding is None:
        return None
    
    try:
        uploaded_image = face_recognition.load_image_file(uploaded_image_path)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)
        
        if not uploaded_encodings:
            return None
        
        face_distance = face_recognition.face_distance([known_encoding], uploaded_encodings[0])[0]
        similarity = (1 - face_distance) * 100
        return max(0, min(100, similarity))
    
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return None