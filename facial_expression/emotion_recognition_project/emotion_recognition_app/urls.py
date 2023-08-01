from django.urls import path
from . import views

urlpatterns = [
    path('', views.emotion_recognition_view, name='default'),
    
]
