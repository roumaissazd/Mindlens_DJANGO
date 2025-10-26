"""
Management command to generate AI-powered reminders for incomplete notes.
Usage: python manage.py generate_note_completion_reminders
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.utils import timezone
from core.models import Notification, Note
from core.ai_utils import analyze_note


class Command(BaseCommand):
    help = 'Génère des rappels IA pour chaque note non terminée des utilisateurs.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Démarrage des rappels IA pour notes incomplètes...'))
        today = timezone.now().date()

        for user in User.objects.all():
            # 1. Trouver toutes les notes non terminées de l'utilisateur
            incomplete_notes = Note.objects.filter(user=user, is_completed=False)

            for note in incomplete_notes:
                # 2. Vérifier si un rappel a déjà été envoyé AUJOURD'HUI pour cette note spécifique
                #    C'est crucial pour ne pas spammer l'utilisateur
                reminder_already_sent = Notification.objects.filter(
                    user=user,
                    timestamp__date=today,
                    message__contains=f"rappel pour votre note : '{note.title or 'Sans titre'}'"
                ).exists()

                if reminder_already_sent:
                    continue  # Passer à la note suivante

                # 3. Analyser le contenu de la note avec l'IA pour un message personnalisé
                analysis = analyze_note(note.content)
                sentiment = analysis.get('sentiment', {}).get('label', 'neutre').lower() if analysis.get('sentiment') else 'neutre'

                # 4. Générer le message de rappel personnalisé
                message = self._generate_reminder_message(note, sentiment)

                # 5. Créer la notification
                Notification.objects.create(user=user, message=message)
                self.stdout.write(self.style.SUCCESS(f'Rappel IA créé pour la note "{note.title or "Sans titre"}" de {user.username}'))

        self.stdout.write(self.style.SUCCESS('Génération des rappels pour notes incomplètes terminée.'))

    def _generate_reminder_message(self, note, sentiment):
        """Génère un message de rappel basé sur le sentiment de la note."""
        title = note.title or 'Sans titre'

        if 'négatif' in sentiment or 'très négatif' in sentiment:
            return f"Un petit mot de soutien. Je pense à votre note '{title}'. J'espère que la situation s'est améliorée. Prenez soin de vous. 💙"
        elif 'positif' in sentiment or 'très positif' in sentiment:
            return f"Rappel pour votre note '{title}' ! C'était une note positive. Avez-vous pu avancer sur ce sujet ? Continuez comme ça ! 😊"
        else:  # Neutre
            return f"Rappel pour votre note '{title}'. Avez-vous eu l'occasion d'y revenir ou de la clôturer ? 🌱"