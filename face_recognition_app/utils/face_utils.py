import face_recognition
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


def compare_faces(uploaded_image_path, targetImage, tolerance=0.4):
    """Compare uploaded image with target image using face recognition.

    Args:
        uploaded_image_path: Path to the uploaded image file
        targetImage: Filename of the target image in media/data/
        tolerance: How much distance between faces to consider it a match (lower is more strict)

    Returns:
        Tuple: (match_result, similarity_score) where:
            match_result: Boolean indicating if faces match
            similarity_score: Percentage similarity (0-100)
    """
    # Load the known face encoding
    known_encoding = load_known_face(targetImage)
    if known_encoding is None:
        return False, 0

    try:
        # Load the uploaded image
        uploaded_image = face_recognition.load_image_file(uploaded_image_path)

        # Find all face locations and encodings in the uploaded image
        face_locations = face_recognition.face_locations(uploaded_image)
        face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

        if not face_encodings:
            return False, 0  # No faces found in uploaded image

        # Compare each face found in the uploaded image with the known face
        for face_encoding in face_encodings:
            # Compare faces
            matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance)

            # Calculate face distance for similarity score
            face_distances = face_recognition.face_distance([known_encoding], face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                # Calculate similarity percentage (0-100)
                similarity = (1 - face_distances[best_match_index]) * 100
                return True, similarity

            # If no matches found, return the highest similarity found
            if face_distances:
                best_similarity = (1 - np.min(face_distances)) * 100
                return False, best_similarity

        return False, 0

    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False, 0