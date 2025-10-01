from django.contrib import admin
from .models import VoiceJournal


class VoiceJournalAdmin(admin.ModelAdmin):
    """Admin configuration for VoiceJournal model."""
    
    list_display = [
        'id',
        'user',
        'transcription_preview',
        'text_sentiment',
        'text_sentiment_score',
        'audio_emotion',
        'audio_emotion_score',
        'is_favorite',
        'created_at',
    ]
    
    list_filter = [
        'text_sentiment',
        'audio_emotion',
        'is_favorite',
        'created_at',
        'user',
    ]
    
    search_fields = [
        'transcription',
        'user__username',
        'user__email',
    ]
    
    readonly_fields = [
        'created_at',
        'updated_at',
        'transcription_preview',
    ]
    
    fieldsets = (
        ('Informations utilisateur', {
            'fields': ('user', 'created_at', 'updated_at')
        }),
        ('Fichier audio', {
            'fields': ('audio_file',)
        }),
        ('Transcription', {
            'fields': ('transcription', 'transcription_preview')
        }),
        ('Analyse du sentiment', {
            'fields': ('text_sentiment', 'text_sentiment_score')
        }),
        ('Détection d\'émotion audio', {
            'fields': ('audio_emotion', 'audio_emotion_score')
        }),
        ('Préférences', {
            'fields': ('is_favorite',)
        }),
    )
    
    ordering = ['-created_at']
    
    def transcription_preview(self, obj):
        """Display truncated transcription in admin list."""
        if obj.transcription:
            return obj.transcription[:100] + "..." if len(obj.transcription) > 100 else obj.transcription
        return "Aucune transcription"
    transcription_preview.short_description = "Transcription (aperçu)"
    
    def get_queryset(self, request):
        """Optimize queries for admin."""
        return super().get_queryset(request).select_related('user')


# Register the model with admin
admin.site.register(VoiceJournal, VoiceJournalAdmin)
