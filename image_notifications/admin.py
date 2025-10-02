from django.contrib import admin

# Register your models here.
from django.db import models
from core.models import Note  # Adjust import based on app name

class GeneratedImage(models.Model):
    note = models.OneToOneField(Note, on_delete=models.CASCADE, related_name='generated_image_obj')
    image_file = models.ImageField(upload_to='generated_images/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.note.title or self.note.created_at.strftime('%d/%m/%Y')}"