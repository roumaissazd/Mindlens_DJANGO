from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.models import Notification, Note
from django.utils import timezone
from datetime import timedelta

class Command(BaseCommand):
    help = 'Génère des notifications de soutien si l utilisateur est inactif, basées sur le sentiment de sa dernière note.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Démarrage de la génération des nudges de soutien...'))
        three_days_ago = timezone.now() - timedelta(days=3)

        for user in User.objects.all():
            # 1. Vérifier si un nudge a déjà été envoyé dans les 3 derniers jours
            if Notification.objects.filter(user=user, timestamp__gte=three_days_ago, message__contains="Je me souviens que").exists():
                continue

            # 2. Trouver la dernière note de l'utilisateur
            last_note = Note.objects.filter(user=user).order_by('-created_at').first()

            # 3. Vérifier si la dernière note est plus ancienne que 3 jours
            if last_note and last_note.created_at < three_days_ago:
                sentiment = last_note.sentiment_label or 'neutre'
                
                # 4. Générer et sauvegarder la notification
                message = self._generate_support_message(sentiment)
                Notification.objects.create(user=user, message=message)
                self.stdout.write(self.style.SUCCESS(f'Nudge de soutien (sentiment: {sentiment}) créé pour {user.username}'))

        self.stdout.write(self.style.SUCCESS('Génération des nudges de soutien terminée.'))

    def _generate_support_message(self, sentiment):
        if 'positif' in sentiment.lower():
            return "Salut ! Cela fait un moment. Je me souviens que votre dernière note était pleine d'énergie positive. J'espère que tout va bien ! 😊"
        elif 'négatif' in sentiment.lower():
            return "Bonjour. Cela fait quelques jours. Je me souviens que votre dernière note n'était pas très joyeuse. J'espère que les choses vont un peu mieux maintenant. Prenez soin de vous. 💙"
        else: # Neutre ou autre
            return "Bonjour ! Cela fait un moment que vous n'avez pas noté vos pensées. Comment vous sentez-vous aujourd'hui ? 🌱"