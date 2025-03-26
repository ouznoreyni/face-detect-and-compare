from django.urls import path
from .views import FaceCompareView, FaceDetectView, WebRTCDetectView, WebRTCFaceMatchAPI

urlpatterns = [
    path('compare/', FaceCompareView.as_view(), name='face-compare'),
    path('detect/', FaceDetectView.as_view(), name='face-detect'),
    path('webrtc-detect/', WebRTCDetectView.as_view(), name='webrtc-detect'),
    path('webrtc-match/', WebRTCFaceMatchAPI.as_view(), name='webrtc-match'),
]
