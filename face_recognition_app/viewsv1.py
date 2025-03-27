import base64
import random

from django.conf import settings
from django.views.generic import TemplateView
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
from .utils.frame_utils import gen_frames


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
        
        # Get target image to test
        targetImage = serializer.validated_data['targetImage']
        # Compare with known face
        similarity = compare_with_known_face(full_path, targetImage)
        
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
        known_encoding = load_known_face("elon.jpg")
        if known_encoding is None:
            return JsonResponse(
                {'error': 'Known face not found in media/data'},
                status=status.HTTP_404_NOT_FOUND
            )

        return StreamingHttpResponse(
            gen_frames(known_encoding),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )




class WebRTCDetectView(TemplateView):
    """Serve the WebRTC face detection page"""
    template_name = "face_detection/webrtc_detect.html"


class WebRTCFaceMatchAPI(APIView):
    """API endpoint to verify face matches via WebRTC"""

    def post(self, request):
        try:
            # Get the image data from the POST request
            image_data = request.data.get('image')
            if not image_data:
                return Response(
                    {'error': 'No image provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate image data format
            if ';base64,' not in image_data:
                return Response(
                    {'error': 'Invalid image format. Expected base64 data URL'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Decode the base64 image data
            try:
                # Extract the base64 part
                header, data = image_data.split(';base64,')
                # Get the image format (jpg, png, etc.)
                img_format = header.split('/')[-1]

                # Decode the image
                image_bytes = base64.b64decode(data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    return Response(
                        {'error': 'Could not decode image'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                # Save the image to disk
                save_dir = os.path.join(settings.MEDIA_ROOT, 'captures')
                os.makedirs(save_dir, exist_ok=True)

                # Generate a unique filename
                filename = f"capture_{random.randint(0, 50)}.{img_format}"
                save_path = os.path.join(save_dir, filename)

                # Write the image file
                cv2.imwrite(save_path, image)

            except Exception as e:
                return Response(
                    {'error': f'Image processing error: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convert to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load known face
            target_image = request.data.get('targetImage', '2.jpeg')
            known_encoding = load_known_face(target_image)
            if known_encoding is None:
                return Response(
                    {'error': f'Known face {target_image} not found in media/data'},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Find faces in the image
            face_locations = face_recognition.face_locations(rgb_image)
            if len(face_locations) != 2:
                return Response({
                    'match': False,
                    'message': 'Exactly two faces required for comparison',
                    'similarity': 0,
                    'face_count': len(face_locations)
                })

            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if len(face_encodings) != 2:
                return Response({
                    'match': False,
                    'message': 'Could not extract features from both faces',
                    'similarity': 0,
                    'face_count': len(face_locations)
                })

            # Compare both faces with the known face
            results = []
            for i, face_encoding in enumerate(face_encodings):
                face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                similarity = float(1 - face_distance) * 100  # Convert to percentage
                match = face_distance < 0.4  # Using standard tolerance

                results.append({
                    'face_index': i,
                    'match': match,
                    'similarity': similarity,
                    'position': {
                        'top': face_locations[i][0],
                        'right': face_locations[i][1],
                        'bottom': face_locations[i][2],
                        'left': face_locations[i][3]
                    }
                })

            # Also compare the two faces against each other
            face_distance_between = face_recognition.face_distance([face_encodings[0]], face_encodings[1])[0]
            similarity_between = float(1 - face_distance_between) * 100
            match_between = face_distance_between < 0.4

            return Response({
                'comparison_with_target': results,
                'face_comparison': {
                    'similarity': similarity_between,
                    'match': match_between,
                    'message': 'Faces match each other' if match_between else 'Faces do not match'
                },
                'face_count': len(face_locations)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()  # Log the full error to console
            return Response(
                {'error': 'Internal server error during face processing'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )