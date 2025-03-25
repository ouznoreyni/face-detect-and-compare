import face_recognition
from deepface import DeepFace
import numpy as np
import os
from django.conf import settings

def load_known_face(targetImage):
    """Load the known face from media/data/known_face.jpg"""
    known_face_path = os.path.join(settings.MEDIA_ROOT, 'data', targetImage)
    if os.path.exists(known_face_path):
        known_image = face_recognition.load_image_file(known_face_path)
        encodings = face_recognition.face_encodings(known_image)
        if encodings:
            return encodings[0]
    return None

def compare_with_known_face(uploaded_image_path, targetImage):
    """Compare uploaded image with known face"""
    known_encoding = load_known_face(targetImage)
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
    


def compare_with_deepface(uploaded_image_path, target_image_path, threshold=0.65):
    """Compares faces using DeepFace (ArcFace model)."""
    try:
        result = DeepFace.verify(
            img1_path=uploaded_image_path,
            img2_path=target_image_path,
            model_name="ArcFace",  # Best for accuracy
            detector_backend="retinaface",  # Better than "opencv"
            enforce_detection=False,  # Skip if no face found
        )
        similarity = (1 - result["distance"]) * 100  # Convert to percentage
        return result["verified"], similarity
    except Exception as e:
        print(f"[DeepFace Error] {e}")
        return False, 0

def compare_faces_hybrid(uploaded_image_path, target_image="known_face.jpg", deepface_threshold=65, fallback_threshold=60):
    """
    Hybrid face comparison:
    1. Tries DeepFace first (high accuracy).
    2. Falls back to face_recognition if DeepFace fails.
    
    Returns:
        - (bool) Whether faces match.
        - (float) Confidence score (0-100).
    """
    target_image_path = os.path.join(settings.MEDIA_ROOT, 'data', target_image)
    
    # Try DeepFace first
    is_match, confidence = compare_with_deepface(uploaded_image_path, target_image_path, deepface_threshold/100)
    
    if confidence >= deepface_threshold:
        return True, confidence
    
    # Fallback to face_recognition if DeepFace fails or confidence is low
    known_encoding = load_known_face(target_image)
    if known_encoding is None:
        return False, 0
    
    try:
        uploaded_image = face_recognition.load_image_file(uploaded_image_path)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)
        
        if not uploaded_encodings:
            return False, 0
        
        face_distance = face_recognition.face_distance([known_encoding], uploaded_encodings[0])[0]
        fallback_confidence = (1 - face_distance) * 100
        
        return (fallback_confidence >= fallback_threshold), fallback_confidence
    
    except Exception as e:
        print(f"[Fallback Error] {e}")
        return False, 0