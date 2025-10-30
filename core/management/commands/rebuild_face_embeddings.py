from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.models import Photo, PhotoAlbum, Profile
from core.face_utils import build_gallery_embeddings, compute_embedding_for_image_path
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Rebuild face embeddings and optimize face recognition'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force rebuild all embeddings',
        )
        parser.add_argument(
            '--user',
            type=str,
            help='Rebuild embeddings for specific user only',
        )

    def handle(self, *args, **options):
        force = options['force']
        target_user = options.get('user')
        
        self.stdout.write(self.style.SUCCESS('=== Rebuilding Face Embeddings ==='))
        
        if target_user:
            try:
                user = User.objects.get(username=target_user)
                photos = Photo.objects.filter(album__user=user)
                profiles = Profile.objects.filter(user=user)
                self.stdout.write(f"Processing user: {user.username}")
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"User {target_user} not found"))
                return
        else:
            photos = Photo.objects.all()
            profiles = Profile.objects.all()
            self.stdout.write("Processing all users")
        
        # Statistiques
        total_photos = photos.count()
        processed_photos = 0
        failed_photos = 0
        
        total_profiles = profiles.exclude(photo_user=None).exclude(photo_user='').count()
        processed_profiles = 0
        failed_profiles = 0
        
        self.stdout.write(f"Found {total_photos} photos and {total_profiles} profile photos")
        
        # Traiter les photos
        for photo in photos:
            if not photo.embedding or force:
                try:
                    self.stdout.write(f"Processing photo {photo.pk}: {photo.person_name}")
                    emb = compute_embedding_for_image_path(photo.image.path)
                    if emb is not None:
                        photo.embedding = emb
                        photo.save(update_fields=['embedding'])
                        processed_photos += 1
                        self.stdout.write(self.style.SUCCESS(f"  ✓ Embedding created ({len(emb)} dimensions)"))
                    else:
                        failed_photos += 1
                        self.stdout.write(self.style.WARNING(f"  ✗ Could not create embedding"))
                except Exception as e:
                    failed_photos += 1
                    self.stdout.write(self.style.ERROR(f"  ✗ Error: {str(e)}"))
            else:
                processed_photos += 1
                self.stdout.write(f"Skipping photo {photo.pk}: already has embedding")
        
        # Traiter les profils
        for profile in profiles.exclude(photo_user=None).exclude(photo_user=''):
            try:
                self.stdout.write(f"Processing profile for {profile.user.username}")
                emb = compute_embedding_for_image_path(profile.photo_user.path)
                if emb:
                    # Créer ou mettre à jour la photo dans l'album __profiles__
                    album, _ = PhotoAlbum.objects.get_or_create(
                        user=profile.user,
                        name='__profiles__',
                        defaults={'description': 'Auto-generated profiles album'}
                    )
                    
                    photo_qs = Photo.objects.filter(album=album, person_name=profile.user.username)
                    if photo_qs.exists():
                        photo = photo_qs.first()
                        photo.embedding = emb
                        photo.save(update_fields=['embedding'])
                    else:
                        Photo.objects.create(
                            album=album,
                            image=profile.photo_user,
                            person_name=profile.user.username,
                            is_user=True,
                            embedding=emb
                        )
                    
                    processed_profiles += 1
                    self.stdout.write(self.style.SUCCESS(f"  ✓ Profile embedding created"))
                else:
                    failed_profiles += 1
                    self.stdout.write(self.style.WARNING(f"  ✗ Could not create profile embedding"))
            except Exception as e:
                failed_profiles += 1
                self.stdout.write(self.style.ERROR(f"  ✗ Error: {str(e)}"))
        
        # Résumé
        self.stdout.write(self.style.SUCCESS('\n=== Summary ==='))
        self.stdout.write(f"Photos: {processed_photos}/{total_photos} processed, {failed_photos} failed")
        self.stdout.write(f"Profiles: {processed_profiles}/{total_profiles} processed, {failed_profiles} failed")
        
        # Vérifier la galerie
        from core.face_utils import _gather_indexed_embeddings
        gallery = _gather_indexed_embeddings()
        self.stdout.write(f"Gallery now contains {len(gallery)} indexed faces")
        
        if gallery:
            self.stdout.write("\nIndexed faces:")
            for item in gallery[:10]:  # Limiter à 10 pour la lisibilité
                self.stdout.write(f"  - {item['person_name']} (photo_id: {item['photo_id']})")
            if len(gallery) > 10:
                self.stdout.write(f"  ... and {len(gallery) - 10} more")
        
        self.stdout.write(self.style.SUCCESS('\nFace embedding rebuild completed!'))
