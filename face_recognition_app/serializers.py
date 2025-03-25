from rest_framework import serializers

class FaceCompareSerializer(serializers.Serializer):
    image = serializers.ImageField()