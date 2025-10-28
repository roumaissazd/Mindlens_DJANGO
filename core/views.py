from urllib import request
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta
import json

from .forms import SignUpForm, NoteForm, SearchForm, ResumeGenerateForm
from .models import Note, Tag, Resume
from .ai_utils import analyze_note, generate_summary_from_notes  
from .search_utils import index_note, remove_note_from_index, search_notes


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
        form = NoteForm(request.POST)
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

            messages.success(request, "Note créée avec succès ! 🎉")
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
        form = NoteForm(request.POST, instance=note)
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

            messages.success(request, "Note mise à jour avec succès ! ✅")
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
@login_required
def resume_generate(request):
    form = ResumeGenerateForm(request.POST or None)
    generated_resume = None

    if request.method == "POST" and form.is_valid():
        period = form.cleaned_data['period']
        category = form.cleaned_data['category']

        notes = Note.objects.filter(user=request.user)

        # Filtrer par période
        now = timezone.now()
        if period == 'week':
            notes = notes.filter(created_at__gte=now - timedelta(days=7))
        elif period == 'month':
            notes = notes.filter(created_at__gte=now - timedelta(days=30))

        # Filtrer par catégorie
        if category:
            notes = notes.filter(category=category)

        # Récupérer le contenu des notes
        notes_contents = [note.content for note in notes]

        if notes_contents:
            summary_text = generate_summary_from_notes(notes_contents)

            # ---- NEW: Create title with date ----
            title = f"Résumé {period}"
            if category:
                title += f" - {category}"
            title += f" ({now.strftime('%d/%m/%Y')})"

            # Sauvegarder le résumé
            generated_resume = Resume.objects.create(
                author=request.user,
                title=title,
                content=summary_text,
                notes_ids=[note.id for note in notes]  # Keep this for future use
            )

    return render(request, "resumes/resume_generate.html", {
        "form": form,
        "generated_resume": generated_resume
    })

@login_required
def resume_list(request):
    """Afficher tous les résumés de l’utilisateur connecté."""
    resumes = Resume.objects.filter(author=request.user).order_by('-created_at')
    return render(request, 'resumes/resume_list.html', {'resumes': resumes})

@login_required
def resume_detail(request, pk):
    """Afficher le détail d'un résumé avec TTS support."""
    resume = get_object_or_404(Resume, pk=pk, author=request.user)
    
    context = {
        'resume': resume,
        # NEW: Pass notes_ids as JSON for potential future use
        'notes_ids_json': resume.notes_ids  # Already a list, will be serialized by Django
    }
    
    return render(request, 'resumes/resume_detail.html', context)