from django.urls import path
from .views import FaceCompareView, FaceDetectView

urlpatterns = [
    path('compare/', FaceCompareView.as_view(), name='face-compare'),
    path('detect/', FaceDetectView.as_view(), name='face-detect'),
]
