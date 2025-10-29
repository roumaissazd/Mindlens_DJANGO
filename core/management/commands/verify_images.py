from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings
import os
import cv2

class Command(BaseCommand):
    help = 'Vérifie les images des utilisateurs et des notes'

    def handle(self, *args, **kwargs):
        # Vérifier les photos de profil
        users = User.objects.filter(profile__photo_user__isnull=False)
        self.stdout.write("Vérification des photos de profil...")
        
        for user in users:
            path = os.path.join(settings.MEDIA_ROOT, str(user.profile.photo_user))
            if os.path.exists(path):
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        self.stdout.write(self.style.SUCCESS(
                            f"✓ Photo de {user.username}: OK ({img.shape})"
                        ))
                    else:
                        self.stdout.write(self.style.ERROR(
                            f"✗ Photo de {user.username}: Impossible de lire l'image"
                        ))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f"✗ Photo de {user.username}: Erreur: {str(e)}"
                    ))
            else:
                self.stdout.write(self.style.ERROR(
                    f"✗ Photo de {user.username}: Fichier non trouvé: {path}"
                ))