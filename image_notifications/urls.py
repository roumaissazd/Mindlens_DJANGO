from django.urls import path
from . import views

urlpatterns = [
    path('generate-image/<int:note_id>/', views.generate_image_for_note, name='generate_image'),
]