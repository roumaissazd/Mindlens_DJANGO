from django.urls import path
from . import views

urlpatterns = [
	path('', views.voice_journal_view, name='voice_journal'),
	path('process-audio/', views.process_audio, name='process_audio'),
]
