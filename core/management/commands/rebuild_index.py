"""
Management command to rebuild the Whoosh search index.
Usage: python manage.py rebuild_index
"""

from django.core.management.base import BaseCommand
from core.models import Note
from core.search_utils import rebuild_index


class Command(BaseCommand):
    help = 'Rebuild the Whoosh search index for all notes'

    def handle(self, *args, **options):
        self.stdout.write('Starting index rebuild...')
        
        notes = Note.objects.all()
        count = rebuild_index(notes)
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully rebuilt index with {count} notes')
        )

