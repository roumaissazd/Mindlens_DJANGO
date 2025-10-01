from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class VoiceJournal(models.Model):
    """Model for storing voice journal entries with AI analysis."""
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='voice_entries'
    )
    
    audio_file = models.FileField(
        upload_to='voice_journal/audio/',
        help_text="Audio file of the voice recording"
    )
    
    transcription = models.TextField(
        blank=True,
        help_text="AI-generated transcription of the audio"
    )
    
    text_sentiment = models.CharField(
        max_length=20,
        blank=True,
        help_text="Sentiment analysis result (POSITIVE, NEGATIVE, NEUTRAL)"
    )
    
    text_sentiment_score = models.FloatField(
        default=0.0,
        help_text="Confidence score for sentiment analysis"
    )
    
    audio_emotion = models.CharField(
        max_length=20,
        blank=True,
        help_text="Detected emotion from audio (calm, excited, angry, sad, neutral)"
    )
    
    audio_emotion_score = models.FloatField(
        default=0.0,
        help_text="Confidence score for emotion detection"
    )
    
    created_at = models.DateTimeField(
        default=timezone.now,
        help_text="When the voice entry was created"
    )
    
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="When the voice entry was last updated"
    )
    
    is_favorite = models.BooleanField(
        default=False,
        help_text="Whether this entry is marked as favorite"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Voice Journal Entry"
        verbose_name_plural = "Voice Journal Entries"
    
    def __str__(self):
        return f"Voice Entry by {self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def duration_display(self):
        """Return formatted duration if available."""
        # This could be enhanced to extract actual audio duration
        return "N/A"
    
    @property
    def transcription_preview(self):
        """Return truncated transcription for display."""
        if len(self.transcription) > 100:
            return self.transcription[:100] + "..."
        return self.transcription
