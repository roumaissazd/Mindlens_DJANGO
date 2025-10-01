from django.apps import AppConfig


class VoiceJournalConfig(AppConfig):
	default_auto_field = 'django.db.models.BigAutoField'
	name = 'voice_journal'
	
	def ready(self):
		"""Import admin configuration when app is ready."""
		import voice_journal.admin