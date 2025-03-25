from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import face_recognition
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from .serializers import FaceCompareSerializer
from .utils.face_utils import load_known_face, compare_with_known_face

class FaceCompareView(APIView):
    def post(self, request):
        """Compare uploaded image with known face in media/data"""
        serializer = FaceCompareSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Save temporary file
        image = serializer.validated_data['image']
        path = default_storage.save(f'tmp/{image.name}', ContentFile(image.read()))
        full_path = default_storage.path(path)
        
        # Compare with known face
        similarity = compare_with_known_face(full_path)
        
        # Clean up
        default_storage.delete(path)
        
        if similarity is None:
            return Response(
                {'error': 'Could not detect faces in one or both images'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response({'similarity_score': similarity})

class FaceDetectView(APIView):
    def get(self, request):
        """Stream webcam video with face recognition against known face"""
        known_encoding = load_known_face()
        if known_encoding is None:
            return JsonResponse(
                {'error': 'Known face not found in media/data'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return StreamingHttpResponse(
            self.gen_frames(known_encoding),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    
    def gen_frames(self, known_encoding):
        """Generate frames with face recognition"""
        video_capture = cv2.VideoCapture(0)
        known_name = "Known Person"
        
        if not video_capture.isOpened():
            yield JsonResponse(
                {'error': 'Could not open webcam'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            return

        process_this_frame = True
        
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            # Only process every other frame to save time
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces([known_encoding], face_encoding)
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance([known_encoding], face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_name
                    
                    face_names.append(name)
            
            process_this_frame = not process_this_frame
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Choose color based on recognition
                if name == known_name:
                    box_color = (0, 255, 0)  # Green
                    text_color = (0, 0, 255)  # Red text
                else:
                    box_color = (0, 0, 255)  # Red
                    text_color = (255, 255, 255)  # White text
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, text_color, 1)
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        video_capture.release()