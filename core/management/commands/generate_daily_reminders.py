from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.models import Notification, Note
from django.utils import timezone
from datetime import timedelta
from collections import Counter
from core.ai_utils import classify_category

class Command(BaseCommand):
    help = 'Génère des rappels quotidiens pour écrire une note, avec contenu IA basé sur les catégories.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Démarrage de la génération des rappels IA quotidiens...'))
        today = timezone.now().date()

        for user in User.objects.all():
            # 1. Vérifier si un rappel a déjà été envoyé aujourd'hui
            if Notification.objects.filter(user=user, timestamp__date=today, message__contains="Pensez à noter").exists():
                continue

            # 2. Récupérer les 5 dernières notes de l'utilisateur
            recent_notes = Note.objects.filter(user=user).order_by('-created_at')[:5]
            if not recent_notes:
                continue

            # 3. Analyser les catégories des notes récentes
            categories = []
            for note in recent_notes:
                # On utilise la catégorie manuelle ou celle générée par l'IA
                category = note.category or (note.auto_tags.get('category_scores') and list(note.auto_tags['category_scores'].keys())[0])
                if category:
                    categories.append(category)
            
            # 4. Trouver la catégorie la plus récurrente
            if categories:
                most_common_category = Counter(categories).most_common(1)[0][0]
            else:
                most_common_category = 'réflexion' # Thème par défaut

            # 5. Générer et sauvegarder la notification
            message = self._generate_reminder_message(most_common_category)
            Notification.objects.create(user=user, message=message)
            self.stdout.write(self.style.SUCCESS(f'Rappel IA (thème: {most_common_category}) créé pour {user.username}'))

        self.stdout.write(self.style.SUCCESS('Génération des rappels IA terminée.'))

    def _generate_reminder_message(self, theme):
        messages = {
            'famille': "Salut ! Pensez à capturer un précieux moment en famille aujourd'hui. Qu'avez-vous partagé de beau ? ❤️",
            'travail': "Bonjour ! Comment s'est passée votre journée professionnelle ? Pensez à noter vos réussites ou vos défis. 💼",
            'voyage': "Bonjour ! Que ce soit un grand voyage ou une petite escapade, notez vos souvenirs d'aventure. ✈️",
            'sante': "Bonjour ! N'oubliez pas de faire le point sur votre bien-être. Comment vous sentez-vous aujourd'hui ? 🌿",
            'amour': "Salut ! L'amour est dans l'air. N'oubliez pas de noter ces moments spéciaux. 💕",
            'loisirs': "Hello ! Un moment de détente ou de loisir aujourd'hui ? C'est le parfait moment pour le noter dans votre journal. 🎮",
            'reflexion': "Bonjour ! Prenez un instant pour la réflexion. Quelle pensée mérite d'être notée aujourd'hui ? 🧠",
            'autre': "Bonjour ! Pensez à noter vos pensées du jour dans votre MindLense. ✍️"
        }
        return messages.get(theme, "Bonjour ! Pensez à noter vos pensées du jour dans votre MindLense. ✍️")