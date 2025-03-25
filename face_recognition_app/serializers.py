from rest_framework import serializers

class FaceCompareSerializer(serializers.Serializer):
    image = serializers.ImageField()
    targetImage = serializers.CharField()