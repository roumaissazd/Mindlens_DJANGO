from core.models import Photo, PhotoAlbum, Profile
from django.contrib.auth.models import User

print('=== UTILISATEURS ===')
for user in User.objects.all():
    print(f'- {user.username}')

print('\n=== PROFILS ===')
for profile in Profile.objects.all():
    has_photo = 'oui' if profile.photo_user else 'non'
    print(f'- {profile.user.username}: photo_user = {has_photo}')
    if profile.photo_user:
        print(f'  Chemin: {profile.photo_user.path}')

print('\n=== ALBUMS PHOTOS ===')
for album in PhotoAlbum.objects.all():
    print(f'- Album "{album.name}" de {album.user.username}: {album.photos.count()} photos')

print('\n=== PHOTOS ===')
for photo in Photo.objects.all():
    has_embedding = 'oui' if photo.embedding else 'non'
    print(f'- {photo.person_name} dans "{photo.album.name}": embedding = {has_embedding}')
    if photo.embedding:
        print(f'  Taille embedding: {len(photo.embedding)}')

print('\n=== VÉRIFICATION GALERIE ===')
from core.face_utils import _gather_indexed_embeddings
gallery = _gather_indexed_embeddings()
print(f'Gallery contient {len(gallery)} embeddings:')
for item in gallery:
    print(f'  - {item["person_name"]} (photo_id: {item["photo_id"]})')
