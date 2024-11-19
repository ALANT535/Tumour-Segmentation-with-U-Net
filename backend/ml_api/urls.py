from django.urls import path
from . import views

urlpatterns = [
    path('api/predict/', views.predict_image, name='predict_image'),
] 