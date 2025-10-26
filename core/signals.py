from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Note, Notification


@receiver(post_save, sender=Note)
def create_note_notification(sender, instance, created, **kwargs):
    """
    Create a notification when a new note is created.
    """
    if created:
        # Create notification for the user
        message = f"Nouvelle note créée : '{instance.title or 'Sans titre'}'"
        if instance.sentiment_label:
            message += f" - Sentiment : {instance.sentiment_label}"

        Notification.objects.create(
            user=instance.user,
            message=message
        )