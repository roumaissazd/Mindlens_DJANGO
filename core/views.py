from django.urls import reverse
from urllib import request
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta
from django.db.models import Q
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.conf import settings
import logging
import json
import numpy as np
from .ai_utils import create_reminder_from_analysis
from .models import Note, Tag, Profile, Reminder, PhotoAlbum, Photo
from .forms import SignUpForm, NoteForm, SearchForm, ProfileForm, PhotoAlbumForm, PhotoForm
from .models import Note, Tag, Profile
from .ai_utils import analyze_note

from .forms import SignUpForm, NoteForm, SearchForm, ProfileForm, ResumeGenerateForm
from .models import Note, Tag, Profile, Resume
from .ai_utils import analyze_note, generate_summary_from_notes, text_to_speech_base64 
from .search_utils import index_note, remove_note_from_index, search_notes
from .face_utils import compare_with_users

logger = logging.getLogger(__name__)
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.core.files.base import ContentFile
from PIL import Image
import base64
import os
from django.urls import reverse


def home(request):
    return render(request, 'home.html')


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, "Bienvenue sur MindLense !")
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', { 'form': form })


def logout(request):
    """Log out the user via GET and redirect to home."""
    auth_logout(request)
    messages.success(request, "À bientôt ! Vous êtes déconnecté(e).")
    return redirect('home')


# ============ NOTES VIEWS ============

@login_required
def note_list(request):
    """Display list of user's notes with filters."""
    notes = Note.objects.filter(user=request.user)

    # Apply filters
    category = request.GET.get('category')
    mood = request.GET.get('mood')
    favorites = request.GET.get('favorites')

    if category:
        notes = notes.filter(category=category)
    if mood:
        notes = notes.filter(mood=mood)
    if favorites:
        notes = notes.filter(is_favorite=True)

    # Get statistics
    stats = {
        'total': Note.objects.filter(user=request.user).count(),
        'favorites': Note.objects.filter(user=request.user, is_favorite=True).count(),
        'this_week': Note.objects.filter(
            user=request.user,
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count(),
    }

    # Get category distribution
    category_stats = Note.objects.filter(user=request.user).exclude(
        category=''
    ).values('category').annotate(count=Count('id'))

    context = {
        'notes': notes,
        'stats': stats,
        'category_stats': category_stats,
        'current_category': category,
        'current_mood': mood,
    }

    return render(request, 'notes/note_list.html', context)


@login_required
def note_detail(request, pk):
    """Display a single note."""
    note = get_object_or_404(Note, pk=pk, user=request.user)

    context = {
        'note': note,
    }

    return render(request, 'notes/note_detail.html', context)


@login_required
def note_create(request):
    """Create a new note."""
    if request.method == 'POST':
        form = NoteForm(request.POST, request.FILES)
        if form.is_valid():
            note = form.save(commit=False)
            note.user = request.user

            # Perform AI analysis
            analysis = analyze_note(note.content)

            # Store sentiment
            if analysis['sentiment']:
                note.sentiment_label = analysis['sentiment']['label']
                note.sentiment_score = analysis['sentiment']['score']

            # Store category if not manually set
            if not note.category and analysis['category']:
                note.category = analysis['category']['category']

            # Store mood if not manually set
            if not note.mood and analysis.get('suggested_mood'):
                note.mood = analysis['suggested_mood']

            # Store auto tags
            note.auto_tags = {
                'tags': analysis['tags'],
                'category_scores': analysis['category'].get('all_scores', {}) if analysis['category'] else {}
            }

            # Save the note first
            note.save()

            # Now handle manual tags (this needs the note to be saved first)
            manual_tags_str = form.cleaned_data.get('manual_tags', '')
            if manual_tags_str:
                tag_names = [name.strip().lower() for name in manual_tags_str.split(',') if name.strip()]

                # Clear existing tags
                note.tags.clear()

                # Add new tags
                for tag_name in tag_names:
                    tag, created = Tag.objects.get_or_create(
                        name=tag_name,
                        defaults={'created_by': note.user, 'is_auto_generated': False}
                    )
                    note.tags.add(tag)

          # Index the note for search
            index_note(note)

            # IA : créer un reminder intelligent
            create_reminder_from_analysis(note, analysis)

            messages.success(request, "Note créée avec succès !")
            return redirect('note_detail', pk=note.pk)

           
    else:
        form = NoteForm()

    context = {
        'form': form,
        'action': 'create',
    }

    return render(request, 'notes/note_form.html', context)


@login_required
def note_edit(request, pk):
    """Edit an existing note."""
    note = get_object_or_404(Note, pk=pk, user=request.user)

    if request.method == 'POST':
        form = NoteForm(request.POST, request.FILES, instance=note)
        if form.is_valid():
            note = form.save(commit=False)

            # Re-analyze if content changed
            if 'content' in form.changed_data:
                analysis = analyze_note(note.content)

                if analysis['sentiment']:
                    note.sentiment_label = analysis['sentiment']['label']
                    note.sentiment_score = analysis['sentiment']['score']

                if not note.category and analysis['category']:
                    note.category = analysis['category']['category']

                # Update mood if not manually set
                if not note.mood and analysis.get('suggested_mood'):
                    note.mood = analysis['suggested_mood']

                note.auto_tags = {
                    'tags': analysis['tags'],
                    'category_scores': analysis['category'].get('all_scores', {}) if analysis['category'] else {}
                }

            # Save the note first
            note.save()

            # Now handle manual tags (this needs the note to be saved first)
            manual_tags_str = form.cleaned_data.get('manual_tags', '')
            if manual_tags_str:
                tag_names = [name.strip().lower() for name in manual_tags_str.split(',') if name.strip()]

                # Clear existing tags
                note.tags.clear()

                # Add new tags
                for tag_name in tag_names:
                    tag, created = Tag.objects.get_or_create(
                        name=tag_name,
                        defaults={'created_by': note.user, 'is_auto_generated': False}
                    )
                    note.tags.add(tag)
            else:
                # If no manual tags provided, clear them
                note.tags.clear()
           
            # Re-index the note
            index_note(note)

            # Recréer le reminder si contenu changé
            if 'content' in form.changed_data:
                create_reminder_from_analysis(note, analysis)

            messages.success(request, "Note mise à jour avec succès !")
            return redirect('note_detail', pk=note.pk)
    else:
        form = NoteForm(instance=note)

    context = {
        'form': form,
        'note': note,
        'action': 'edit',
    }

    return render(request, 'notes/note_form.html', context)


@login_required
def note_delete(request, pk):
    """Delete a note."""
    note = get_object_or_404(Note, pk=pk, user=request.user)

    if request.method == 'POST':
        # Remove from search index
        remove_note_from_index(note.id)

        note.delete()
        messages.success(request, "Note supprimée avec succès.")
        return redirect('note_list')

    context = {
        'note': note,
    }

    return render(request, 'notes/note_confirm_delete.html', context)


@login_required
def note_toggle_favorite(request, pk):
    """Toggle favorite status of a note (AJAX)."""
    if request.method == 'POST':
        note = get_object_or_404(Note, pk=pk, user=request.user)
        note.is_favorite = not note.is_favorite
        note.save()

        return JsonResponse({
            'success': True,
            'is_favorite': note.is_favorite
        })

    return JsonResponse({'success': False}, status=400)


@login_required
def note_search(request):
    """Search notes with filters."""
    form = SearchForm(request.GET)
    results = []
    query_text = None

    if form.is_valid():
        query_text = form.cleaned_data.get('q')
        category = form.cleaned_data.get('category')
        mood = form.cleaned_data.get('mood')
        tags_str = form.cleaned_data.get('tags')

        # Parse tags
        tags = None
        if tags_str:
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]

        # Search using Whoosh
        if query_text or category or mood or tags:
            note_ids = search_notes(
                user_id=request.user.id,
                query_text=query_text,
                category=category,
                mood=mood,
                tags=tags
            )

            # Get notes in the order returned by search
            if note_ids:
                notes_dict = {note.id: note for note in Note.objects.filter(id__in=note_ids)}
                results = [notes_dict[note_id] for note_id in note_ids if note_id in notes_dict]
        else:
            # No search criteria, show all notes
            results = Note.objects.filter(user=request.user)

    context = {
        'form': form,
        'results': results,
        'query': query_text,
    }

    return render(request, 'notes/search_results.html', context)


@login_required
def dashboard(request):
    """Display dashboard with statistics and charts."""
    user_notes = Note.objects.filter(user=request.user)

    # Basic stats
    stats = {
        'total_notes': user_notes.count(),
        'favorites': user_notes.filter(is_favorite=True).count(),
        'this_week': user_notes.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count(),
        'this_month': user_notes.filter(
            created_at__gte=timezone.now() - timedelta(days=30)
        ).count(),
    }

    # Category distribution (sorted by count descending)
    category_stats = list(user_notes.exclude(
        category=''
    ).values('category').annotate(count=Count('id')).order_by('-count'))

    # Mood distribution (sorted by count descending)
    mood_stats = list(user_notes.exclude(
        mood=''
    ).values('mood').annotate(count=Count('id')).order_by('-count'))

    # Recent notes
    recent_notes = user_notes[:5]

    # Sentiment distribution
    sentiment_stats = []
    if user_notes.filter(sentiment_label__isnull=False).exists():
        sentiment_stats = list(user_notes.exclude(
            sentiment_label=''
        ).values('sentiment_label').annotate(count=Count('id')))

    # Activity over time (last 30 days)
    activity_data = []
    for i in range(30, -1, -1):
        date = timezone.now().date() - timedelta(days=i)
        count = user_notes.filter(created_at__date=date).count()
        activity_data.append({
            'date': date.strftime('%d/%m'),
            'count': count
        })

    context = {
        'stats': stats,
        'category_stats': category_stats,
        'mood_stats': mood_stats,
        'sentiment_stats': sentiment_stats,
        'recent_notes': recent_notes,
        'activity_data': json.dumps(activity_data),
    }

    return render(request, 'notes/dashboard.html', context)


@login_required
def note_export_json(request):
    """Export all user notes as JSON."""
    notes = Note.objects.filter(user=request.user)

    data = []
    for note in notes:
        data.append({
            'id': note.id,
            'title': note.title,
            'content': note.content,
            'mood': note.mood,
            'category': note.category,
            'tags': [tag.name for tag in note.tags.all()],
            'auto_tags': note.get_auto_tags_list(),
            'sentiment': note.sentiment_label,
            'is_favorite': note.is_favorite,
            'created_at': note.created_at.isoformat(),
            'updated_at': note.updated_at.isoformat(),
        })

    response = HttpResponse(
        json.dumps(data, indent=2, ensure_ascii=False),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="mindlens_notes_{timezone.now().strftime("%Y%m%d")}.json"'

    return response


@login_required
def detect_faces_in_note(request, pk):
    """Détecte les visages dans l'image d'une note"""
    note = get_object_or_404(Note, pk=pk, user=request.user)
    
    if not note.image:
        return JsonResponse({'error': 'Cette note ne contient pas d\'image'}, status=400)
    
    logger.info(f"=== Détection des visages pour la note {pk} ===")
    logger.info(f"Chemin de l'image: {note.image.path}")
    logger.info(f"URL de l'image: {note.image.url}")
    logger.info(f"Titre de la note: {note.title}")
    
    try:
        # Vérifier si le fichier image existe
        if not os.path.exists(note.image.path):
            logger.error(f"Fichier image introuvable: {note.image.path}")
            return JsonResponse({'error': 'Fichier image introuvable'}, status=404)
        
        # Obtenir la taille du fichier pour diagnostic
        file_size = os.path.getsize(note.image.path)
        logger.info(f"Taille du fichier image: {file_size} bytes")
        
        # Lancer la détection
        results = compare_with_users(note.image.path)
        logger.info(f"Résultats bruts de compare_with_users: {results}")
        
        # Analyser les résultats
        if not results:
            logger.info("Aucun visage détecté dans l'image")
            return JsonResponse({
                'success': True,
                'faces': [],
                'message': 'Aucun visage détecté dans cette image'
            })
        
        # Compter les visages identifiés vs inconnus
        identified_count = sum(1 for r in results if r.get('username'))
        unknown_count = len(results) - identified_count
        
        logger.info(f"Détection terminée: {len(results)} visages trouvés ({identified_count} identifiés, {unknown_count} inconnus)")
        
        # Préparer la réponse détaillée
        response_data = {
            'success': True,
            'faces': results,
            'summary': {
                'total_faces': len(results),
                'identified_faces': identified_count,
                'unknown_faces': unknown_count
            }
        }
        
        # Ajouter des informations sur chaque visage
        for i, face in enumerate(results):
            if face.get('username'):
                logger.info(f"Visage {i+1}: {face['username']} (confiance: {face['confidence']}%)")
            else:
                logger.info(f"Visage {i+1}: Inconnu (position: {face['location']})")
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.exception(f"Erreur lors de la détection des visages pour la note {pk}: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Erreur lors de la détection: {str(e)}',
            'error_type': type(e).__name__
        }, status=500)


@login_required
def build_face_gallery(request):
    """Génère les embeddings pour tous les visages existants (profiles et photos)"""
    from .face_utils import compute_embedding_for_image_path
    
    results = {
        'profiles_processed': 0,
        'profiles_failed': 0,
        'photos_processed': 0,
        'photos_failed': 0,
        'photos_skipped': 0,
        'details': []
    }
    
    try:
        logger.info("=== Début de la construction de la galerie de visages ===")
        
        # Traiter les profils utilisateurs EN PRIORITÉ
        profiles = Profile.objects.exclude(photo_user=None).exclude(photo_user='')
        logger.info(f"Traitement de {profiles.count()} profils utilisateurs")
        
        for profile in profiles:
            try:
                if profile.photo_user and profile.photo_user.path:
                    logger.info(f"Traitement du profil: {profile.user.username}")
                    emb = compute_embedding_for_image_path(profile.photo_user.path)
                    
                    if emb:
                        # Créer ou mettre à jour dans l'album __profiles__
                        album, _ = PhotoAlbum.objects.get_or_create(
                            user=profile.user,
                            name='__profiles__',
                            defaults={'description': 'Album automatique des profils'}
                        )
                        
                        # Supprimer les anciennes entrées pour éviter les doublons
                        Photo.objects.filter(album=album, person_name=profile.user.username).delete()
                        
                        # Créer la nouvelle photo avec embedding
                        Photo.objects.create(
                            album=album,
                            image=profile.photo_user,
                            person_name=profile.user.username,
                            is_user=True,
                            embedding=emb
                        )
                        
                        results['profiles_processed'] += 1
                        results['details'].append(f"✓ Profil {profile.user.username}: embedding créé (priorité haute)")
                        logger.info(f"✓ Profil {profile.user.username}: embedding créé avec succès")
                    else:
                        results['profiles_failed'] += 1
                        results['details'].append(f"✗ Profil {profile.user.username}: échec embedding")
                        logger.warning(f"✗ Profil {profile.user.username}: échec de création d'embedding")
            except Exception as e:
                results['profiles_failed'] += 1
                results['details'].append(f"✗ Profil {profile.user.username}: erreur {str(e)[:50]}")
                logger.exception(f"Erreur lors du traitement du profil {profile.user.username}: {e}")
        
        # Traiter les photos des albums (sauf __profiles__)
        photos = Photo.objects.exclude(album__name='__profiles__')
        logger.info(f"Traitement de {photos.count()} photos d'albums")
        
        for photo in photos:
            try:
                # Vérifier si le nom de fichier contient des caractères spéciaux problématiques
                if photo.image and photo.image.path:
                    filename = os.path.basename(photo.image.path)
                    if any(char in filename for char in ['é', 'è', 'ê', 'à', 'ù', 'ç', 'â', 'î', 'ô', 'û']):
                        logger.warning(f"Photo {photo.pk} ({filename}) contient des caractères spéciaux, traitement spécial")
                    
                    # Toujours recalculer l'embedding pour s'assurer qu'il est à jour
                    emb = compute_embedding_for_image_path(photo.image.path)
                    if emb:
                        photo.embedding = emb
                        photo.save(update_fields=['embedding'])
                        results['photos_processed'] += 1
                        results['details'].append(f"✓ Photo {photo.person_name}: embedding créé")
                        logger.info(f"✓ Photo {photo.pk} ({photo.person_name}): embedding créé")
                    else:
                        results['photos_failed'] += 1
                        results['details'].append(f"✗ Photo {photo.person_name}: échec embedding")
                        logger.warning(f"✗ Photo {photo.pk} ({photo.person_name}): échec embedding")
            except Exception as e:
                results['photos_failed'] += 1
                results['details'].append(f"✗ Photo {photo.person_name}: erreur {str(e)[:50]}")
                logger.exception(f"Erreur lors du traitement de la photo {photo.pk}: {e}")
        
        # Vérifier le résultat final
        from .face_utils import _gather_indexed_embeddings
        gallery = _gather_indexed_embeddings()
        results['total_embeddings'] = len(gallery)
        
        # Compter les profils vs photos d'album
        profile_count = sum(1 for g in gallery if g['priority'] == 1)
        album_count = sum(1 for g in gallery if g['priority'] == 2)
        
        logger.info(f"=== Galerie construite: {len(gallery)} embeddings ({profile_count} profils, {album_count} albums) ===")
        
        return JsonResponse({
            'success': True,
            'results': results,
            'gallery_summary': {
                'total': len(gallery),
                'profiles': profile_count,
                'albums': album_count
            },
            'gallery_faces': [{'name': g['person_name'], 'id': g['photo_id'], 'priority': g['priority']} for g in gallery[:15]]
        })
        
    except Exception as e:
        logger.exception(f"Erreur lors de la construction de la galerie: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required
def detect_faces_in_album_photo(request, photo_pk):
    """Détecte les visages dans une photo d'album et retourne les coordonnées pour tagging"""
    from .face_utils import _try_deepface_extract_faces
    
    photo = get_object_or_404(Photo, pk=photo_pk, album__user=request.user)
    
    try:
        if not photo.image or not photo.image.path:
            return JsonResponse({'error': 'Photo sans image'}, status=400)
        
        logger.info(f"Détection des visages dans la photo {photo_pk}")
        
        # Détecter les visages
        faces = _try_deepface_extract_faces(photo.image.path)
        
        if not faces:
            return JsonResponse({
                'success': True,
                'faces': [],
                'message': 'Aucun visage détecté'
            })
        
        # Formater les résultats
        face_data = []
        for i, face in enumerate(faces):
            facial_area = face.get('facial_area', {})
            face_data.append({
                'index': i,
                'location': {
                    'x': facial_area.get('x', 0),
                    'y': facial_area.get('y', 0),
                    'w': facial_area.get('w', 0),
                    'h': facial_area.get('h', 0)
                },
                'confidence': face.get('confidence', 0)
            })
        
        logger.info(f"Détecté {len(face_data)} visages dans la photo {photo_pk}")
        
        return JsonResponse({
            'success': True,
            'faces': face_data,
            'photo_id': photo.pk
        })
        
    except Exception as e:
        logger.exception(f"Erreur lors de la détection des visages dans la photo {photo_pk}: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required
def tag_face_in_photo(request, photo_pk):
    """Extrait un visage d'une photo et crée une nouvelle entrée Photo avec ce visage"""
    from .face_utils import compute_embedding_for_image_path
    from PIL import Image
    import io
    from django.core.files.uploadedfile import InMemoryUploadedFile
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)
    
    photo = get_object_or_404(Photo, pk=photo_pk, album__user=request.user)
    
    try:
        # Récupérer les données du visage
        person_name = request.POST.get('person_name', '').strip()
        x = int(request.POST.get('x', 0))
        y = int(request.POST.get('y', 0))
        w = int(request.POST.get('w', 0))
        h = int(request.POST.get('h', 0))
        
        if not person_name:
            return JsonResponse({'error': 'Nom de personne requis'}, status=400)
        
        if w <= 0 or h <= 0:
            return JsonResponse({'error': 'Coordonnées invalides'}, status=400)
        
        logger.info(f"Extraction du visage de {person_name} depuis la photo {photo_pk}")
        
        # Charger l'image originale
        img = Image.open(photo.image.path).convert('RGB')
        
        # Extraire le visage avec un padding de 20%
        padding = int(max(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.width, x + w + padding)
        y2 = min(img.height, y + h + padding)
        
        face_img = img.crop((x1, y1, x2, y2))
        
        # Sauvegarder le visage dans un buffer
        buffer = io.BytesIO()
        face_img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Créer un fichier uploadé
        face_file = InMemoryUploadedFile(
            buffer,
            'ImageField',
            f'face_{person_name}_{photo_pk}.jpg',
            'image/jpeg',
            buffer.getbuffer().nbytes,
            None
        )
        
        # Calculer l'embedding du visage extrait
        face_array = np.array(face_img)
        embedding = compute_embedding_for_image_path(face_array)
        
        if not embedding:
            logger.warning(f"Impossible de calculer l'embedding pour {person_name}")
        
        # Créer une nouvelle photo avec ce visage
        new_photo = Photo.objects.create(
            album=photo.album,
            image=face_file,
            person_name=person_name,
            is_user=False,
            embedding=embedding
        )
        
        logger.info(f"Visage de {person_name} extrait et sauvegardé (photo {new_photo.pk})")
        
        return JsonResponse({
            'success': True,
            'photo_id': new_photo.pk,
            'person_name': person_name,
            'has_embedding': embedding is not None
        })
        
    except Exception as e:
        logger.exception(f"Erreur lors de l'extraction du visage: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# ============ PROFILE VIEWS ============

@login_required
def profile_view(request):
    # Créer le profil s'il n'existe pas
    Profile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = ProfileForm(
            request.POST, 
            request.FILES, 
            instance=request.user.profile
        )
        
        if form.is_valid():
            # Mettre à jour le profil
            form.save()
            
            # Mettre à jour les informations de l'utilisateur
            user = request.user
            user.username = request.POST.get('username')
            user.email = request.POST.get('email')
            user.save()
            
            messages.success(request, "Profil mis à jour avec succès !")
            return redirect('profile')
    else:
        form = ProfileForm(instance=request.user.profile)

    return render(request, 'profile.html', {
        'form': form,
        'user': request.user
    })
@login_required
def api_unread_reminders(request):
    reminders = Reminder.objects.filter(
        user=request.user,
        is_read=False,
        trigger_at__lte=timezone.now()
    ).order_by('-priority', 'trigger_at')[:8]

    data = {
        'count': reminders.count(),
        'reminders': [
            {
                'id': r.id,
                'message': r.message,
                'priority': r.priority,
                'url': reverse('note_detail', args=[r.note.pk]),
                'time': r.trigger_at.strftime('%H:%M')
            }
            for r in reminders
        ]
    }
    return JsonResponse(data)
 #Resume
# ============ RESUMES VIEWS ============

@login_required
def resume_generate(request):
    form = ResumeGenerateForm(request.POST or None)
    generated_resume = None

    if request.method == "POST" and form.is_valid():
        period = form.cleaned_data['period']
        category = form.cleaned_data['category']

        notes = Note.objects.filter(user=request.user)

        now = timezone.now()
        if period == 'week':
            notes = notes.filter(created_at__gte=now - timedelta(days=7))
        elif period == 'month':
            notes = notes.filter(created_at__gte=now - timedelta(days=30))
        if category:
            notes = notes.filter(category=category)

        notes_contents = [note.content for note in notes]

        if notes_contents:
            summary_text = generate_summary_from_notes(notes_contents)

            # Titre simple et clair (sans IA)
            period_label = "la semaine" if period == "week" else "le mois"
            category_label = f" ({category})" if category else ""
            date_str = now.strftime("%d/%m/%Y")
            final_title = f"Résumé de {period_label}{category_label} - {date_str}"

            generated_resume = Resume.objects.create(
                author=request.user,
                title=final_title,
                content=summary_text,
                notes_ids=[note.id for note in notes]
            )

    return render(request, "resumes/resume_generate.html", {
        "form": form,
        "generated_resume": generated_resume
    })


@login_required
def resume_list(request):
    resumes = Resume.objects.filter(author=request.user).order_by('-created_at')

    current_period = request.GET.get('period')
    current_category = request.GET.get('category')

    if current_period:
        now = timezone.now()
        if current_period == 'week':
            resumes = resumes.filter(created_at__gte=now - timedelta(days=7))
        elif current_period == 'month':
            resumes = resumes.filter(created_at__gte=now - timedelta(days=30))
        elif current_period == 'year':
            resumes = resumes.filter(created_at__year=now.year)

    if current_category:
        resumes = resumes.filter(category=current_category)

    stats = {
        'total': Resume.objects.filter(author=request.user).count(),
        'this_week': Resume.objects.filter(author=request.user, created_at__gte=timezone.now() - timedelta(days=7)).count(),
        'this_month': Resume.objects.filter(author=request.user, created_at__gte=timezone.now() - timedelta(days=30)).count(),
    }

    context = {
        'resumes': resumes,
        'stats': stats,
        'current_period': current_period,
        'current_category': current_category,
    }
    return render(request, 'resumes/resume_list.html', context)


@login_required
def resume_detail(request, pk):
    resume = get_object_or_404(Resume, pk=pk, author=request.user)

    if not resume.audio_b64:
        resume.audio_b64 = text_to_speech_base64(resume.content)
        resume.save(update_fields=['audio_b64'])

    return render(request, 'resumes/resume_detail.html', {
        'resume': resume,
        'audio_b64': resume.audio_b64,
    })


@login_required
def resume_edit(request, pk):
    resume = get_object_or_404(Resume, pk=pk, author=request.user)

    if request.method == "POST":
        new_content = request.POST.get("content", "").strip()
        if not new_content:
            messages.error(request, "Le résumé ne peut pas être vide.")
        else:
            resume.content = new_content
            resume.audio_b64 = text_to_speech_base64(new_content)
            resume.save()
            messages.success(request, "Résumé mis à jour avec succès !")
        return redirect('resume_detail', pk=resume.pk)

    return render(request, "resumes/resume_edit.html", {"resume": resume})


@login_required
def resume_delete(request, pk):
    resume = get_object_or_404(Resume, pk=pk, author=request.user)
    if request.method == "POST":
        resume.delete()
        messages.success(request, "Résumé supprimé avec succès.")
        return redirect('resume_list')
    return redirect('resume_list')


@login_required
def resume_toggle_favorite(request, pk):
    if request.method != 'POST':
        return JsonResponse({'success': False}, status=400)

    resume = get_object_or_404(Resume, pk=pk, author=request.user)
    resume.is_favorite = not resume.is_favorite
    resume.save()
    return JsonResponse({'success upholstery': True, 'is_favorite': resume.is_favorite})


@login_required
def resume_search(request):
    query = request.GET.get('q', '').strip()
    period = request.GET.get('period', '')
    category = request.GET.get('category', '')

    resumes = Resume.objects.filter(author=request.user)

    if period:
        now = timezone.now()
        if period == 'week':
            resumes = resumes.filter(created_at__gte=now - timedelta(days=7))
        elif period == 'month':
            resumes = resumes.filter(created_at__gte=now - timedelta(days=30))
        elif period == 'year':
            resumes = resumes.filter(created_at__year=now.year)

    if category:
        resumes = resumes.filter(category=category)

    results = resumes.distinct().order_by('-created_at')

    categories = [
        ('famille', 'Famille'), ('travail', 'Travail'), ('voyage', 'Voyage'),
        ('sante', 'Santé'), ('amour', 'Amour'), ('loisirs', 'Loisirs'), ('reflexion', 'Réflexion')
    ]

    return render(request, 'resumes/resume_search.html', {
        'results': results,
        'query': query,
        'period': period,
        'category': category,
        'categories': categories,
    })

# ============ PHOTOS VIEWS ============

@login_required
def photo_album_list(request):
    albums = PhotoAlbum.objects.filter(user=request.user)
    return render(request, 'photos/album_list.html', {'albums': albums})

@login_required
def photo_album_create(request):
    if request.method == 'POST':
        form = PhotoAlbumForm(request.POST)
        if form.is_valid():
            album = form.save(commit=False)
            album.user = request.user
            album.save()
            messages.success(request, "Album créé avec succès !")
            return redirect('photo_album_detail', pk=album.pk)
    else:
        form = PhotoAlbumForm()
    return render(request, 'photos/album_form.html', {'form': form, 'action': 'create'})


@login_required
def photo_album_detail(request, pk):
    album = get_object_or_404(PhotoAlbum, pk=pk, user=request.user)
    photos_qs = album.photos.all().order_by('-created_at')

    # Search by person name
    q = request.GET.get('q', '').strip()
    if q:
        photos_qs = photos_qs.filter(person_name__icontains=q)

    # Pagination
    paginator = Paginator(photos_qs, 20)  # 20 par page
    page_number = request.GET.get('page')
    photos_page = paginator.get_page(page_number)

    if request.method == 'POST':
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            photo = form.save(commit=False)
            photo.album = album
            # is_user check is done in model.save but we can set explicitly
            photo.is_user = User.objects.filter(username=photo.person_name).exists()
            photo.save()
            messages.success(request, "Photo ajoutée avec succès !")
            return redirect('photo_album_detail', pk=album.pk)
    else:
        form = PhotoForm()

    users = User.objects.values_list('username', flat=True)
    
    return render(request, 'photos/album_detail.html', {
        'album': album,
        'photos': photos_page,
        'form': form,
        'users': users,
        'q': q,
        'paginator': paginator,
        'page_obj': photos_page,
    })


@login_required
def photo_album_edit(request, pk):
    album = get_object_or_404(PhotoAlbum, pk=pk, user=request.user)
    if request.method == 'POST':
        form = PhotoAlbumForm(request.POST, instance=album)
        if form.is_valid():
            form.save()
            messages.success(request, "Album modifié avec succès !")
            return redirect('photo_album_detail', pk=pk)
    else:
        form = PhotoAlbumForm(instance=album)
    return render(request, 'photos/album_form.html', {
        'form': form,
        'album': album,
        'action': 'edit'
    })

@login_required
def photo_album_delete(request, pk):
    album = get_object_or_404(PhotoAlbum, pk=pk, user=request.user)
    if request.method == 'POST':
        album.delete()
        messages.success(request, "Album supprimé avec succès !")
        return redirect('photo_album_list')
    return render(request, 'photos/album_confirm_delete.html', {'album': album})

@login_required
def photo_delete(request, album_pk, photo_pk):
    photo = get_object_or_404(Photo, pk=photo_pk, album__pk=album_pk, album__user=request.user)
    if request.method == 'POST':
        photo.delete()
        messages.success(request, "Photo supprimée avec succès !")
    return redirect('photo_album_detail', pk=album_pk)

@login_required
@require_POST
def photo_create_from_face(request):
    """
    AJAX endpoint: create a cropped Photo from a face bbox in a note image.
    Expects: note_pk, top, right, bottom, left, optional person_name
    """
    note_pk = request.POST.get('note_pk')
    top = request.POST.get('top')
    right = request.POST.get('right')
    bottom = request.POST.get('bottom')
    left = request.POST.get('left')
    person_name = request.POST.get('person_name', '').strip()

    if not all([note_pk, top, right, bottom, left]):
        return JsonResponse({'success': False, 'error': 'Missing parameters'}, status=400)

    try:
        from .models import Note  # local import to avoid cycle
        note = get_object_or_404(Note, pk=int(note_pk), user=request.user)
        if not note.image:
            return JsonResponse({'success': False, 'error': 'Note has no image'}, status=400)

        src_path = note.image.path
        top, right, bottom, left = int(float(top)), int(float(right)), int(float(bottom)), int(float(left))

        # Ensure user has a training album
        album, _ = PhotoAlbum.objects.get_or_create(user=request.user, name='Training Data', defaults={'description': 'Photos sélectionnées pour entraînement automatique'})

        # Crop with PIL
        img = Image.open(src_path).convert("RGB")
        crop = img.crop((left, top, right, bottom))

        # Build filename and save to media
        filename = f"crop_note{note.pk}_{int(timezone.now().timestamp())}.jpg"
        save_path = os.path.join(settings.MEDIA_ROOT, 'album_photos', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        crop.save(save_path, format='JPEG', quality=90)

        # Create Photo record referencing saved file (use relative path for ImageField)
        rel_path = os.path.join('album_photos', filename)
        photo = Photo.objects.create(album=album, image=rel_path, person_name=person_name or 'unknown', is_user=bool(person_name and User.objects.filter(username=person_name).exists()))

        # Optionally compute embedding now (if face_utils is available)
        try:
            from .face_utils import compute_embedding_for_image_path
            emb = compute_embedding_for_image_path(photo.image.path)
            if emb:
                photo.embedding = emb
                photo.save(update_fields=['embedding'])
        except Exception:
            pass

        return JsonResponse({
            'success': True,
            'photo_id': photo.pk,
            'person_name': photo.person_name,
            'photo_url': photo.image.url
        })
    except Exception as e:
        logger.exception("Error creating face photo: %s", e)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
