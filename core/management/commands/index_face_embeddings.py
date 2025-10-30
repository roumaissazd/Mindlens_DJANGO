from django.core.management.base import BaseCommand
from core.face_utils import build_gallery_embeddings

class Command(BaseCommand):
    help = "Compute embeddings for all photos and profile images (build face gallery)."

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true', help='Force re-compute of all embeddings')

    def handle(self, *args, **options):
        force = options['force']
        build_gallery_embeddings(force=force)
        self.stdout.write(self.style.SUCCESS("Done indexing face embeddings"))