from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
from core.models import Note
from .models import GeneratedImage
from .utils import generate_image_from_note
import logging
import timezone

logger = logging.getLogger(__name__)

def daily_reminder_task():
    for user in User.objects.all():
        if not Note.objects.filter(user=user, created_at__date=timezone.now().date()):
            send_mail(
                'Mindlens Reminder',
                'Time to add a note today and generate an image!',
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
            )
            logger.info(f"Reminder sent to {user.email}")

def check_image_generation_task():
    for note in Note.objects.filter(generated_image_obj__isnull=True):
        if generate_image_from_note(note):
            GeneratedImage.objects.update_or_create(
                note=note,
                defaults={'image_file': generate_image_from_note(note)}
            )
            send_mail(
                'Mindlens Update',
                f'An image has been generated for your note "{note.title or note.content[:20]}"!',
                settings.DEFAULT_FROM_EMAIL,
                [note.user.email],
            )
            logger.info(f"Notification sent for note {note.id}")