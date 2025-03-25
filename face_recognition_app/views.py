
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
from .utils.face_utils import compare_faces_hybrid, load_known_face, compare_with_known_face
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
        # similarity = compare_with_known_face(full_path, targetImage)
        
        # # Clean up
        # default_storage.delete(path)
        
        # if similarity is None:
        #     return Response(
        #         {'error': 'Could not detect faces in one or both images'},
        #         status=status.HTTP_400_BAD_REQUEST
        #     )
        
        # return Response({'similarity_score': similarity})

        # Compare faces using hybrid method
        is_match, confidence = compare_faces_hybrid(full_path, targetImage)
         # # Clean up
        default_storage.delete(path)
        return JsonResponse({
            "status": "success",
            "match": is_match,
            "confidence": confidence,
            "method": "DeepFace + face_recognition (Hybrid)",
        })

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
