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
from .ai_utils import translate_note_mymemory
from .ai_utils import generate_title
from rest_framework.decorators import api_view
from rest_framework.response import Response

import json
import logging
from .ai_utils import create_reminder_from_analysis
from .models import Note, Tag, Profile, Reminder
from .forms import SignUpForm, NoteForm, SearchForm, ProfileForm
from .models import Note, Tag, Profile
from .ai_utils import analyze_note
from .ai_utils import generate_smart_advice
from .forms import SignUpForm, NoteForm, SearchForm, ProfileForm, ResumeGenerateForm
from .models import Note, Tag, Profile, Resume
from .ai_utils import analyze_note, generate_summary_from_notes, text_to_speech_base64 
from .search_utils import index_note, remove_note_from_index, search_notes
from .face_utils import compare_with_users

logger = logging.getLogger(__name__)
from django.contrib import messages


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
    
    logger.info(f"Détection des visages pour la note {pk}")
    logger.info(f"Chemin de l'image: {note.image.path}")
    logger.info(f"URL de l'image: {note.image.url}")
    
    try:
        results = compare_with_users(note.image.path)
        logger.info(f"Résultats trouvés: {results}")
        
        return JsonResponse({
            'success': True,
            'faces': results
        })
    except Exception as e:
        logger.error(f"Erreur lors de la détection: {str(e)}")
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

@login_required
def translate_note_view(request, pk):
    note = get_object_or_404(Note, pk=pk, user=request.user)
    
    if request.method == 'POST':
        target = request.POST.get('target_language', 'en')
        success = translate_note_mymemory(note, target)
        if success:
            messages.success(request, f"Traduit en {target.upper()} !")
        else:
            messages.error(request, "Erreur traduction (MyMemory).")
    
    return redirect('note_detail', pk=note.pk)
    # core/views.py (ajoute cette vue)
@login_required
def notification_list(request):
    """Liste complète des notifications (rappels) de l'utilisateur."""
    reminders = Reminder.objects.filter(user=request.user).order_by('-priority', '-trigger_at')
    
    context = {
        'reminders': reminders,
        'title': 'Toutes les Notifications',
    }
    return render(request, 'notes/notification_list.html', context)


@login_required
def delete_reminder(request, pk):
    """Supprime une notification (AJAX)."""
    if request.method == 'POST':
        reminder = get_object_or_404(Reminder, pk=pk, user=request.user)
        reminder.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

@api_view(['POST'])
def api_generate_title(request):
    text = request.data.get('content', '').strip()
    if not text:
        return Response({"error": "Contenu requis"}, status=400)
    
    title = generate_title(text)
    return Response({"title": title})




def note_detail(request, pk):
    note = Note.objects.get(pk=pk)
    
    if not note.smart_advice and note.content:
        note.smart_advice = generate_smart_advice(note.content)
        note.save()

    return render(request, 'notes/note_detail.html', {'note': note})    