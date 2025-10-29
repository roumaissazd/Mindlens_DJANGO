from django.urls import path
from . import views

urlpatterns = [
	path('', views.voice_journal_view, name='voice_journal'),
	path('list/', views.voice_journal_list, name='voice_journal_list'),
	path('<int:pk>/', views.voice_journal_detail, name='voice_journal_detail'),
	path('<int:pk>/delete/', views.voice_journal_delete, name='voice_journal_delete'),
	path('<int:pk>/toggle-favorite/', views.voice_journal_toggle_favorite, name='voice_journal_toggle_favorite'),
	path('process-audio/', views.process_audio, name='process_audio'),
	path('api/analyze-voice/', views.api_analyze_voice, name='api_analyze_voice'),
	path('api/voice-agent-turn/', views.api_voice_agent_turn, name='api_voice_agent_turn'),
	path('api/music/', views.api_music_recommendations, name='api_music'),
]

