from django.apps import AppConfig
from django.core.management import call_command
import threading
import time


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        import core.signals  # noqa

        # Démarrer les tâches planifiées en arrière-plan
        def run_scheduled_tasks():
            while True:
                try:
                    # Exécuter generate_support_nudges à 10h00
                    current_time = time.strftime("%H:%M")
                    if current_time == "10:00":
                        call_command('generate_support_nudges', verbosity=1)

                    # Exécuter generate_note_completion_reminders à 11h00
                    elif current_time == "11:00":
                        call_command('generate_note_completion_reminders', verbosity=1)

                    # Attendre 60 secondes avant de vérifier à nouveau
                    time.sleep(60)

                except Exception as e:
                    print(f"Erreur dans les tâches planifiées: {e}")
                    time.sleep(60)

        # Démarrer le thread en arrière-plan (seulement en production)
        import os
        if os.environ.get('RUN_MAIN') != 'true':  # Éviter le double démarrage en dev
            thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
            thread.start()
