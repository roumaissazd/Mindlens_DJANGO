from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json
from django.db.models.signals import post_save
from django.dispatch import receiver



class Tag(models.Model):
    """Tag model for categorizing notes."""
    name = models.CharField(max_length=50, unique=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_tags'
    )
    is_auto_generated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name





class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    photo_user = models.ImageField(upload_to='user_photos/', blank=True, null=True)

    def __str__(self):
        return self.user.username







class Note(models.Model):
    """Note model for storing user journal entries."""

    MOOD_CHOICES = [
        ('joyeux', '😊 Joyeux'),
        ('triste', '😢 Triste'),
        ('neutre', '😐 Neutre'),
        ('anxieux', '😰 Anxieux'),
        ('excite', '🤩 Excité'),
        ('calme', '😌 Calme'),
        ('colere', '😠 En colère'),
    ]

    CATEGORY_CHOICES = [
        ('famille', 'Famille'),
        ('travail', 'Travail'),
        ('voyage', 'Voyage'),
        ('sante', 'Santé'),
        ('amour', 'Amour'),
        ('loisirs', 'Loisirs'),
        ('reflexion', 'Réflexion'),
        ('autre', 'Autre'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notes')
    title = models.CharField(max_length=200, blank=True)
    content = models.TextField()
    image = models.ImageField(upload_to='notes_images/', blank=True, null=True, verbose_name='Image') # Ajout du champ image
    mood = models.CharField(max_length=20, choices=MOOD_CHOICES, blank=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, blank=True)

    # AI-generated metadata
    auto_tags = models.JSONField(default=dict, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)
    sentiment_label = models.CharField(max_length=50, blank=True)

    # Manual tags
    tags = models.ManyToManyField(Tag, blank=True, related_name='notes')

    # Features
    is_favorite = models.BooleanField(default=False)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['category']),
            models.Index(fields=['mood']),
        ]

    def __str__(self):
        return self.title or f"Note du {self.created_at.strftime('%d/%m/%Y')}"

    def get_auto_tags_list(self):
        """Return auto tags as a list."""
        if isinstance(self.auto_tags, dict):
            return self.auto_tags.get('tags', [])
        return []

    def get_sentiment_emoji(self):
        """Return emoji based on sentiment."""
        if not self.sentiment_label:
            return '😐'

        sentiment_emojis = {
            'très positif': '🤩',
            'positif': '😊',
            'neutre': '😐',
            'négatif': '😔',
            'très négatif': '😢',
        }
        return sentiment_emojis.get(self.sentiment_label.lower(), '😐')

    def get_mood_emoji(self):
        """Return emoji for the mood."""
        mood_emojis = {
            'joyeux': '😊',
            'triste': '😢',
            'neutre': '😐',
            'anxieux': '😰',
            'excite': '🤩',
            'calme': '😌',
            'colere': '😠',
        }
        return mood_emojis.get(self.mood, '😐')

    def get_category_icon(self):
        """Return icon for the category."""
        category_icons = {
            'famille': '👨‍👩‍👧‍👦',
            'travail': '💼',
            'voyage': '✈️',
            'sante': '🏥',
            'amour': '❤️',
            'loisirs': '🎮',
            'reflexion': '🧠',
            'autre': '📝',
        }
        return category_icons.get(self.category, '📝')


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


class Reminder(models.Model):
    PRIORITY_CHOICES = [
        ('haute', 'Haute'),
        ('moyenne', 'Moyenne'),
        ('basse', 'Basse'),
    ]

    note = models.ForeignKey(Note, on_delete=models.CASCADE, related_name='reminders')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES)
    trigger_at = models.DateTimeField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-priority', '-trigger_at']

    def __str__(self):
        return f"{self.get_priority_display()} - {self.message[:30]}"