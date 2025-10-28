import os
import re
import tempfile
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.conf import settings
import requests
from .models import VoiceJournal
import time
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# LLM for intelligent replies
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False
    print("Groq not available. Install with: pip install groq")

# Import ML libraries with error handling (spaces only)
ML_LIBRARIES_AVAILABLE = True
ML_MISSING_MODULES: list[str] = []
try:
    import io
    import torch  # type: ignore
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"torch ({e})")
try:
    from transformers import pipeline  # type: ignore
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"transformers ({e})")
    pipeline = None  # type: ignore
try:
    import librosa  # type: ignore
    import numpy as np  # type: ignore
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"librosa/numpy ({e})")
try:
    from pydub import AudioSegment  # type: ignore
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"pydub ({e})")
if not ML_LIBRARIES_AVAILABLE:
    print(f"Warning: ML libraries not available: {', '.join(ML_MISSING_MODULES)}")

# Control whether to use heavyweight sentiment model (defaults to off for speed)
USE_SENTIMENT_MODEL = os.environ.get('USE_SENTIMENT_MODEL', '0') in {'1', 'true', 'yes'}


# ------------------------
# AI helper utilities
# ------------------------

EMOTION_TO_QUERY = {
    'happy': 'upbeat background music',
    'sad': 'soothing piano music',
    'calm': 'ambient chill music',
    'stressed': 'relaxing breathing music',
    'neutral': 'lofi background music'
}

EMOTION_TO_THEME = {
    'happy':   {'primary': '#50E3C2', 'secondary': '#F5A623'},
    'sad':     {'primary': '#5DADE2', 'secondary': '#C39BD3'},
    'calm':    {'primary': '#BDC3C7', 'secondary': '#FDF2E9'},
    'stressed':{'primary': '#F5A623', 'secondary': '#5DADE2'},
    'neutral': {'primary': '#F9FAFB', 'secondary': '#F9FAFB'},
}

MOOD_FALLBACK_VIDEOS = {
    # Reliable, embeddable IDs (instrumental/background-friendly)
    'happy':   ['DWcJFNfaw9c', 'jfKfPfyJRdk', '7NOSDKb0HlU'],
    'sad':     ['1ZYbU82GVz4', '2OEL4P1Rz04', '1fUEoXz0Q0g'],
    'calm':    ['7NOSDKb0HlU', 'DWcJFNfaw9c', 'jfKfPfyJRdk'],
    'stressed':['2OEL4P1Rz04', '1ZYbU82GVz4', 'jfKfPfyJRdk'],
    'neutral': ['DWcJFNfaw9c', 'jfKfPfyJRdk', '7NOSDKb0HlU'],
}

def generate_ai_message(emotion: str) -> str:
    mapping = {
        'happy':   "That’s wonderful! Keep that positive energy going ✨",
        'sad':     "I can feel some sadness in your voice… want to talk more about it?",
        'calm':    "You sound peaceful today 🌿. It’s nice to hear that.",
        'stressed':"Let’s take a deep breath together. Want to record a few thoughts to clear your mind?",
        'neutral': "Got it! Would you like to add more details or just save it as a memory?",
    }
    return mapping.get(emotion, mapping['neutral'])


def decide_mood(audio_emotion: str, text_sentiment: str) -> str:
    """
    Combine audio emotion + text sentiment to a final mood label.
    Privilégie le sentiment du texte quand il y a une contradiction claire.
    """
    sentiment_map = {
        'POSITIVE': 'happy',
        'NEGATIVE': 'sad',
        'NEUTRAL':  'neutral'
    }
    
    normalized_audio = (audio_emotion or '').lower().strip()
    normalized_sentiment = (text_sentiment or '').upper().strip()
    
    # Si le sentiment du texte est clair, l'utiliser en priorité
    if normalized_sentiment in sentiment_map:
        text_mood = sentiment_map[normalized_sentiment]
        
        # Vérifier s'il y a une contradiction majeure
        contradictions = {
            ('sad', 'excited'): True,
            ('sad', 'happy'): True,
            ('happy', 'sad'): True,
            ('happy', 'angry'): True,
        }
        
        if contradictions.get((text_mood, normalized_audio), False):
            # En cas de contradiction, privilégier le sentiment du texte
            print(f"[Mood] Contradiction détectée: texte={text_mood}, audio={normalized_audio}. Privilégiant le texte.")
            return text_mood
    
    # Sinon, utiliser l'émotion audio si elle est valide
    if normalized_audio in {'happy', 'sad', 'calm', 'stressed', 'neutral', 'angry', 'excited'}:
        return normalized_audio
    
    # Fallback sur le sentiment du texte
    return sentiment_map.get(normalized_sentiment, 'neutral')


def get_youtube_music_recommendations(emotion: str) -> list[dict]:
    """Return up to 3 YouTube video recs.
    If YOUTUBE_API_KEY is set in env or settings, use Data API v3; otherwise, return search links.
    """
    query = EMOTION_TO_QUERY.get(emotion, 'lofi background music')
    api_key = os.environ.get('YOUTUBE_API_KEY', getattr(settings, 'YOUTUBE_API_KEY', ''))
    results: list[dict] = []
    if api_key:
        try:
            params = {
                'part': 'snippet',
                'q': query,
                'maxResults': 3,
                'type': 'video',
                'key': api_key,
                'safeSearch': 'moderate'
            }
            resp = requests.get('https://www.googleapis.com/youtube/v3/search', params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            for item in (data.get('items') or [])[:3]:
                vid = item.get('id', {}).get('videoId')
                snippet = item.get('snippet', {})
                if not vid:
                    continue
                results.append({
                    'title': snippet.get('title', 'Music'),
                    'url': f'https://www.youtube.com/watch?v={vid}',
                    'thumbnail': (snippet.get('thumbnails', {}).get('medium', {}) or {}).get('url'),
                    'videoId': vid,
                    'embedUrl': f'https://www.youtube.com/embed/{vid}'
                })
        except Exception:
            pass

    if not results:
        # Fallback: curated embeds by mood (no API key required)
        vids = MOOD_FALLBACK_VIDEOS.get(emotion) or MOOD_FALLBACK_VIDEOS['neutral']
        for vid in vids[:3]:
            results.append({
                'title': 'Recommended music',
                'url': f'https://www.youtube.com/watch?v={vid}',
                'thumbnail': None,
                'videoId': vid,
                'embedUrl': f'https://www.youtube-nocookie.com/embed/{vid}?rel=0&modestbranding=1&iv_load_policy=3'
            })
    return results[:3]

# Initialize models (cache them to avoid reloading)
_asr_pipeline = None
_sentiment_pipeline = None


def get_asr_pipeline():
    """Get or initialize automatic speech recognition (Whisper via HF)."""
    if not ML_LIBRARIES_AVAILABLE:
        raise ImportError("ML libraries not available")
    global _asr_pipeline
    
    # Use cached pipeline if available
    if _asr_pipeline is not None:
        return _asr_pipeline
    
    try:
        # Uses Hugging Face Whisper model locally with PyTorch backend
        # Small/base models are faster and lighter
        print("[ASR] Creating Whisper pipeline... (this may take a while on first use)")
        _asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=-1,  # Use CPU (avoiding GPU dependencies)
            chunk_length_s=30,
            return_timestamps=False,
            generate_kwargs={
                'task': 'transcribe',
                'language': getattr(settings, 'ASR_LANGUAGE', 'fr')
            },
        )
        print("[ASR] Whisper pipeline created successfully")
    except Exception as e:
        print(f"[ASR] Error creating pipeline: {e}")
        # Reset global variable on error
        _asr_pipeline = None
        raise e
    
    return _asr_pipeline


def get_sentiment_pipeline():
    """Get or initialize multilingual sentiment analysis pipeline.
    Uses a model that supports FR/EN and returns negative/neutral/positive.
    """
    if not ML_LIBRARIES_AVAILABLE:
        raise ImportError("ML libraries not available")
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            # Robust multilingual sentiment model
            _sentiment_pipeline = pipeline(
                task="text-classification",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                top_k=None,
            )
        except Exception:
            # Fallback to default if model not available
            _sentiment_pipeline = pipeline("sentiment-analysis")
    return _sentiment_pipeline


def normalize_sentiment_label(raw_label: str, score: float) -> tuple[str, float]:
    """Normalize various model label styles to (POSITIVE|NEGATIVE|NEUTRAL, score)."""
    label = (raw_label or '').strip().lower()
    mapping = {
        'positive': 'POSITIVE',
        'neutral': 'NEUTRAL',
        'negative': 'NEGATIVE',
        'label_2': 'POSITIVE',
        'label_1': 'NEUTRAL',
        'label_0': 'NEGATIVE',
        '5 stars': 'POSITIVE',
        '4 stars': 'POSITIVE',
        '3 stars': 'NEUTRAL',
        '2 stars': 'NEGATIVE',
        '1 star': 'NEGATIVE',
    }
    return mapping.get(label, 'NEUTRAL'), float(score or 0.0)


def heuristic_fr_sentiment(text: str) -> tuple[str, float]:
    """Very fast FR heuristic for POSITIVE/NEGATIVE based on keywords with basic negation.
    Returns (label, confidence in 0..1). NEUTRAL if no strong signal.
    """
    t = (text or '').lower()
    if not t:
        return 'NEUTRAL', 0.0
    positives = {
        'heureux', 'heureuse', 'content', 'contente', 'joyeux', 'joyeuse', 'bien', 'super', 'ravi', 'ravie', 'calme', 'apaisé', 'apaisee', 'apaisée'
    }
    negatives = {
        'triste', 'mal', 'deprime', 'déprimé', 'déprime', 'deprimé', 'stress', 'stresse', 'stressé', 'angoisse', 'angoissé', 'peur', 'fatigué', 'fatigue', 'fatiguée'
    }
    neutral_words = {
        'neutre', 'ni'
    }
    # Basic tokenization
    for ch in [',', '.', '!', '?', ':', ';', '…', '’', '"', '“', '”']:
        t = t.replace(ch, ' ')
    words = t.split()
    # Detect neutral pattern "ni ... ni ..."
    if ' ni ' in f' {t} ' and t.count(' ni ') >= 2:
        return 'NEUTRAL', 0.7
    score = 0
    for i, w in enumerate(words):
        prev = words[i-1] if i > 0 else ''
        negated = (prev == 'pas' or prev == "n'" or prev == 'ne')
        if w in positives:
            score += (-1 if negated else 1)
        if w in negatives:
            score += (1 if negated else -1)
        if w in neutral_words:
            # slight pull towards neutral
            score += 0
    if score >= 1:
        return 'POSITIVE', min(1.0, 0.6 + 0.1 * score)
    if score <= -1:
        return 'NEGATIVE', min(1.0, 0.6 + 0.1 * (-score))
    return 'NEUTRAL', 0.6 if ('neutre' in words) else 0.0


def generate_empathetic_reply(final_mood: str, user_text: str) -> str:
    """Return a short, empathetic reply tailored to the mood and what the user said."""
    text = (user_text or '').strip()
    mood = (final_mood or 'neutral').lower()
    if mood in {'sad', 'stressed'}:
        opener = "Je t'entends."
        if mood == 'sad':
            core = (
                "Tu sembles traverser un moment difficile. "
                "On peut prendre une minute pour respirer ensemble: inspire 4 secondes, bloque 2, expire 6. "
                "Si tu veux, dis-moi ce qui pèse le plus en ce moment."
            )
        else:
            core = (
                "Tu sembles tendu(e). Essayons une respiration 4-7-8: inspire 4, bloque 7, expire 8. "
                "Dis-moi ce qui te stresse le plus et ce que tu as déjà essayé."
            )
        tip = (
            "Petit rappel: ce que tu ressens est valable. On peut avancer pas à pas."
        )
        if text:
            return f"{opener} {core} J'ai entendu: “{text[:160]}…”. {tip}"
        return f"{opener} {core} {tip}"
    if mood == 'happy':
        return "Super d'entendre cela ✨. Qu'est-ce qui te rend le plus joyeux aujourd'hui?"
    if mood == 'calm':
        return "Tu sembles apaisé(e). Veux-tu ancrer ce moment avec une intention pour la journée?"
    return "Merci pour le partage. Dis-m'en un peu plus pour que je puisse t'aider."


def generate_opening_prompt(final_mood: str) -> str:
    mood = (final_mood or 'neutral').lower()
    if mood == 'sad':
        return "Bonjour, je suis là pour t'écouter. Dis-moi ce que tu ressens ou ce dont tu as besoin."
    if mood == 'stressed':
        return "Salut, je comprends que tu sois stressé(e). Parle-moi de ce qui se passe ou ce que tu aimerais faire."
    if mood == 'happy':
        return "Bonjour ! Ça fait plaisir d'entendre ça. Que veux-tu partager ?"
    if mood == 'calm':
        return "Hello, ambiance apaisée. Que veux-tu faire ? On peut discuter, écouter de la musique, ou autre chose."
    return "Bonjour ! Je suis là pour toi. Dis-moi ce que tu ressens ou ce dont tu as besoin."


def generate_llm_reply(user_text: str, mood: str, conversation_history: list = None) -> str:
    """
    Generate intelligent reply using Groq LLM (free).
    Falls back to rule-based if Groq not available.
    """
    # Get API key from settings
    api_key = getattr(settings, 'GROQ_API_KEY', '')
    
    print(f"[LLM] Checking Groq: available={GROQ_AVAILABLE}, has_key={bool(api_key)}")
    
    if not GROQ_AVAILABLE or not api_key:
        print("[LLM] Groq not available, using fallback")
        # Fallback to rule-based
        return generate_empathetic_reply(mood, user_text)
    
    try:
        client = Groq(api_key=api_key)
        
        # Build context
        context = f"""Tu es un assistant vocal empathique et naturel. L'utilisateur semble être {mood}.

Réponds de façon:
- Naturelle, comme à un ami
- Courte (1-2 phrases max 20 mots)
- Empathique et chaleureux
- Personnalisée selon ce qu'il dit

Tu peux:
- L'écouter et répondre avec des mots qui tiennent
- Poser des questions
- Donner des conseils si il en demande
- Proposer de la musique seulement si il mentionne vouloir écouter
- Proposer une respiration seulement si il est très stressé

Reste naturel, pas robotique. Réponds comme tu penserais si c'était un ami qui te parle."""

        messages = [
            {"role": "system", "content": context}
        ]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-4:])  # Keep last 4 exchanges
        
        # Add current user message
        messages.append({"role": "user", "content": user_text})
        
        # Generate response
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Free, fast model
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Post-processing to keep it short
        if len(reply) > 80:
            # Try to cut at a sentence boundary
            sentences = reply.split('. ')
            if len(sentences) > 1:
                reply = sentences[0] + '.'
            else:
                reply = reply[:80] + '...'
        
        return reply
        
    except Exception as e:
        print(f"[LLM Error] {e}")
        # Fallback to rule-based
        return generate_empathetic_reply(mood, user_text)


def detect_user_intent_yes_no(text: str) -> str:
    """Very simple yes/no detector for short confirmations in FR/EN.
    Returns 'yes', 'no', or 'unknown'.
    """
    t = (text or '').strip().lower()
    if not t:
        return 'unknown'
    # Normalize punctuation
    for ch in [',', '.', '!', '?', ':', ';', '…', '’', '\'', '"', '“', '”', '’']:
        t = t.replace(ch, ' ')
    words = set(t.split())
    yes_words = {
        'oui', 'ouais', 'ouii', 'ouiii', 'yes', 'yep', 'yeah', 'certainement', 'ok', 'okay', 'daccord', 'd\'accord', 'si', 'ui'
    }
    no_words = {
        'non', 'no', 'nope', 'nan', 'nop', 'nonmerci', 'non-merci'
    }
    # Also handle phrases containing clear yes/no
    if any(w in words for w in yes_words) or ' oui ' in f' {t} ' or t.startswith('oui'):
        return 'yes'
    if any(w in words for w in no_words) or ' non ' in f' {t} ' or t.startswith('non') or t.startswith('no'):
        return 'no'
    # Heuristics for like/dislike in FR
    if re.search(r"\bj[' ]?aime\b", t):
        # If contains direct negation around "aime"
        if re.search(r"\b(n[' ]?e?)?\s*aime\s*pas\b", t) or 'aime pas' in t:
            return 'no'
        return 'yes'
    if 'change' in t or 'changer' in t or 'autre' in t or 'suivant' in t or 'next' in t or 'skip' in t:
        return 'no'
    if 'oui merci' in t:
        return 'yes'
    if 'non merci' in t:
        return 'no'
    # Heuristic for very short utterances
    if t in {'oui.', 'oui!', 'yes.', 'yes!'}:
        return 'yes'
    if t in {'non.', 'non!', 'no.', 'no!'}:
        return 'no'
    return 'unknown'


def analyze_audio_emotion(audio_path):
    """
    Analyze emotion from audio using librosa features
    This is a simplified emotion detection based on audio features
    """
    if not ML_LIBRARIES_AVAILABLE:
        return "neutral", 0.5
        
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract audio features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate statistics
        mfcc_mean = np.mean(mfccs, axis=1)
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Simple emotion classification based on features
        # This is a basic heuristic - in production, you'd use a trained model
        
        if spectral_centroid_mean > 2000 and zcr_mean > 0.1:
            emotion = "excited"
            score = 0.8
        elif spectral_centroid_mean < 1000 and zcr_mean < 0.05:
            emotion = "calm"
            score = 0.7
        elif zcr_mean > 0.15:
            emotion = "angry"
            score = 0.6
        elif spectral_centroid_mean < 1200:
            emotion = "sad"
            score = 0.6
        else:
            emotion = "neutral"
            score = 0.5
            
        return emotion, score
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return "neutral", 0.5


def convert_to_wav(audio_file):
    """Convert audio file to WAV format"""
    if not ML_LIBRARIES_AVAILABLE:
        return None
        
    try:
        # Detect input format by extension and read
        ext = os.path.splitext(str(audio_file))[-1].lower().lstrip('.')
        fmt = None
        if ext in {"wav", "webm", "ogg", "mp3", "m4a", "aac", "flac"}:
            fmt = ext
        # Read audio file (pydub/ffmpeg)
        if fmt == 'wav':
            audio = AudioSegment.from_wav(audio_file)
        elif fmt:
            audio = AudioSegment.from_file(audio_file, format=fmt)
        else:
            audio = AudioSegment.from_file(audio_file)
        
        # Convert to WAV format
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        return wav_buffer
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None


@login_required
def voice_journal_view(request):
    """Main voice journal page - requires login"""
    return render(request, 'voice_journal/voice_journal.html')


@login_required
def voice_journal_list(request):
    """List all voice journal entries for the current user with filters"""
    voice_entries = VoiceJournal.objects.filter(user=request.user)
    
    # Get filter parameters
    search_query = request.GET.get('search', '').strip()
    sentiment_filter = request.GET.get('sentiment', '')
    emotion_filter = request.GET.get('emotion', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    favorites_only = request.GET.get('favorites', '')
    
    # Apply filters
    if search_query:
        voice_entries = voice_entries.filter(transcription__icontains=search_query)
    
    if sentiment_filter:
        # Normalize the filter value for comparison
        clean_sentiment_filter = sentiment_filter.strip().upper()
        voice_entries = voice_entries.filter(text_sentiment__iexact=clean_sentiment_filter)
    
    if emotion_filter:
        # Normalize the filter value for comparison
        clean_emotion_filter = emotion_filter.strip().lower()
        voice_entries = voice_entries.filter(audio_emotion__iexact=clean_emotion_filter)
    
    if date_from:
        voice_entries = voice_entries.filter(created_at__date__gte=date_from)
    
    if date_to:
        voice_entries = voice_entries.filter(created_at__date__lte=date_to)
    
    if favorites_only:
        voice_entries = voice_entries.filter(is_favorite=True)
    
    # Order by creation date (newest first)
    voice_entries = voice_entries.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(voice_entries, 10)  # 10 entries per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get unique values for filter dropdowns with labels
    unique_sentiments_raw = VoiceJournal.objects.filter(user=request.user).values_list('text_sentiment', flat=True).distinct()
    unique_emotions_raw = VoiceJournal.objects.filter(user=request.user).values_list('audio_emotion', flat=True).distinct()
    
    # Create sentiment and emotion mappings for better display
    SENTIMENT_LABELS = {
        'POSITIVE': '😊 Positif',
        'NEGATIVE': '😔 Négatif', 
        'NEUTRAL': '😐 Neutre'
    }
    
    EMOTION_LABELS = {
        'happy': '😊 Joyeux',
        'sad': '😢 Triste',
        'calm': '😌 Calme',
        'stressed': '😰 Stressé',
        'angry': '😠 En colère',
        'excited': '🤩 Excité',
        'neutral': '😐 Neutre'
    }
    
    # Prepare sentiment choices with labels (deduplicate and clean)
    unique_sentiments = []
    seen_sentiments = set()
    for sentiment in unique_sentiments_raw:
        if sentiment:  # Skip empty values
            # Clean and normalize the sentiment value
            clean_sentiment = sentiment.strip().upper()
            if clean_sentiment not in seen_sentiments:
                seen_sentiments.add(clean_sentiment)
                unique_sentiments.append({
                    'value': clean_sentiment,
                    'label': SENTIMENT_LABELS.get(clean_sentiment, clean_sentiment)
                })
    
    # Sort sentiments in logical order
    sentiment_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    unique_sentiments.sort(key=lambda x: sentiment_order.index(x['value']) if x['value'] in sentiment_order else 999)
    
    # Prepare emotion choices with labels (deduplicate and clean)
    unique_emotions = []
    seen_emotions = set()
    for emotion in unique_emotions_raw:
        if emotion:  # Skip empty values
            # Clean and normalize the emotion value
            clean_emotion = emotion.strip().lower()
            if clean_emotion not in seen_emotions:
                seen_emotions.add(clean_emotion)
                unique_emotions.append({
                    'value': clean_emotion,
                    'label': EMOTION_LABELS.get(clean_emotion, clean_emotion)
                })
    
    # Sort emotions in logical order
    emotion_order = ['happy', 'excited', 'calm', 'neutral', 'sad', 'stressed', 'angry']
    unique_emotions.sort(key=lambda x: emotion_order.index(x['value']) if x['value'] in emotion_order else 999)
    
    # Get display labels for active filters
    clean_sentiment_filter = sentiment_filter.strip().upper() if sentiment_filter else ''
    clean_emotion_filter = emotion_filter.strip().lower() if emotion_filter else ''
    
    active_sentiment_label = SENTIMENT_LABELS.get(clean_sentiment_filter, clean_sentiment_filter) if clean_sentiment_filter else ''
    active_emotion_label = EMOTION_LABELS.get(clean_emotion_filter, clean_emotion_filter) if clean_emotion_filter else ''
    
    context = {
        'page_obj': page_obj,
        'voice_entries': page_obj,
        'search_query': search_query,
        'sentiment_filter': sentiment_filter,
        'emotion_filter': emotion_filter,
        'date_from': date_from,
        'date_to': date_to,
        'favorites_only': favorites_only,
        'unique_sentiments': unique_sentiments,
        'unique_emotions': unique_emotions,
        'active_sentiment_label': active_sentiment_label,
        'active_emotion_label': active_emotion_label,
    }
    return render(request, 'voice_journal/voice_journal_list.html', context)


@login_required
def voice_journal_detail(request, pk):
    """View details of a specific voice journal entry"""
    try:
        voice_entry = VoiceJournal.objects.get(pk=pk, user=request.user)
    except VoiceJournal.DoesNotExist:
        messages.error(request, "Cet enregistrement vocal n'existe pas.")
        return redirect('voice_journal_list')
    
    context = {
        'voice_entry': voice_entry,
    }
    return render(request, 'voice_journal/voice_journal_detail.html', context)


@login_required
def voice_journal_delete(request, pk):
    """Delete a voice journal entry"""
    try:
        voice_entry = VoiceJournal.objects.get(pk=pk, user=request.user)
        voice_entry.delete()
        messages.success(request, "L'enregistrement vocal a été supprimé avec succès.")
    except VoiceJournal.DoesNotExist:
        messages.error(request, "Cet enregistrement vocal n'existe pas.")
    
    return redirect('voice_journal_list')


@login_required
def voice_journal_toggle_favorite(request, pk):
    """Toggle favorite status of a voice journal entry"""
    try:
        voice_entry = VoiceJournal.objects.get(pk=pk, user=request.user)
        voice_entry.is_favorite = not voice_entry.is_favorite
        voice_entry.save()
        
        status = "ajouté aux favoris" if voice_entry.is_favorite else "retiré des favoris"
        messages.success(request, f"L'enregistrement vocal a été {status}.")
    except VoiceJournal.DoesNotExist:
        messages.error(request, "Cet enregistrement vocal n'existe pas.")
    
    return redirect('voice_journal_list')


@csrf_exempt
@require_http_methods(["POST"])
@login_required
def process_audio(request):
    """Process uploaded audio file and save to database"""
    if not ML_LIBRARIES_AVAILABLE:
        return JsonResponse({'error': 'ML libraries not available. Please install required packages.', 'missing': ML_MISSING_MODULES}, status=500)

    try:
        if 'audio' not in request.FILES:
            return JsonResponse({'error': 'No audio file provided'}, status=400)
        
        audio_file = request.FILES['audio']
        
        # Save audio file temporarily
        temp_path = default_storage.save(f'temp/{audio_file.name}', ContentFile(audio_file.read()))
        full_path = default_storage.path(temp_path)
        
        try:
            # Convert to WAV if necessary
            t_conv0 = time.time()
            wav_buffer = convert_to_wav(full_path)
            if wav_buffer is None:
                return JsonResponse({'error': 'Failed to convert audio'}, status=500)
            t_conv1 = time.time()
            
            # Save WAV file with user-specific naming
            user_tag = request.user.id
            wav_filename = f"voice_journal_{user_tag}_{temp_path.split('/')[-1]}.wav"
            wav_path = default_storage.save(f'voice_journal/audio/{wav_filename}', ContentFile(wav_buffer.getvalue()))
            wav_full_path = default_storage.path(wav_path)
            
            # Transcribe audio using Whisper (Hugging Face transformers)
            t_asr0 = time.time()
            transcription = ""
            try:
                print(f"[ASR] Starting transcription of {wav_full_path}")
                asr = get_asr_pipeline()
                asr_result = asr(wav_full_path, return_timestamps=False)
                transcription = asr_result.get("text", "").strip()
                if transcription:
                    print(f"[ASR] Transcription successful: '{transcription[:50]}...'")
                else:
                    print("[ASR] Transcription returned empty string")
            except Exception as e:
                print(f"[ASR] Error during transcription: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback - use audio emotion for mood detection instead
                transcription = "[Enregistrement audio non transcrit]"
                print("[ASR] Using fallback transcription")
            t_asr1 = time.time()
            
            # Analyze text sentiment
            t_sent0 = time.time()
            # Heuristic first; optionally fallback to model if enabled
            text_sentiment, text_sentiment_score = heuristic_fr_sentiment(transcription)
            if USE_SENTIMENT_MODEL and text_sentiment == 'NEUTRAL':
                sentiment_pipeline = get_sentiment_pipeline()
                sentiment_result = sentiment_pipeline(transcription)[0]
                text_sentiment, text_sentiment_score = normalize_sentiment_label(
                    sentiment_result.get('label', ''),
                    float(sentiment_result.get('score', 0.0))
                )
            t_sent1 = time.time()
            
            # Analyze audio emotion
            t_em0 = time.time()
            audio_emotion, audio_emotion_score = analyze_audio_emotion(wav_full_path)
            t_em1 = time.time()
            
            # Save to database
            voice_entry = VoiceJournal.objects.create(
                user=request.user,
                audio_file=wav_path,
                transcription=transcription,
                text_sentiment=text_sentiment,
                text_sentiment_score=text_sentiment_score,
                audio_emotion=audio_emotion,
                audio_emotion_score=audio_emotion_score
            )
            
            # Compose enriched AI output
            final_mood = decide_mood(audio_emotion, text_sentiment)
            ai_message = generate_ai_message(final_mood)
            theme = EMOTION_TO_THEME.get(final_mood, EMOTION_TO_THEME['neutral'])
            music = get_youtube_music_recommendations(final_mood)

            # Clean up temporary file
            default_storage.delete(temp_path)
            
            return JsonResponse({
                'success': True,
                'transcription': transcription,
                'text_sentiment': text_sentiment,
                'text_sentiment_score': text_sentiment_score,
                'audio_emotion': audio_emotion,
                'audio_emotion_score': audio_emotion_score,
                'audio_url': default_storage.url(wav_path),
                'emotion': final_mood,
                'ai_message': ai_message,
                'theme': theme,
                'music': music,
                'entry_id': voice_entry.id,
                'timings_ms': {
                    'convert': int((t_conv1 - t_conv0) * 1000),
                    'asr': int((t_asr1 - t_asr0) * 1000),
                    'sentiment': int((t_sent1 - t_sent0) * 1000),
                    'emotion': int((t_em1 - t_em0) * 1000)
                }
            })
            
        except Exception as e:
            # Clean up temporary file on error
            default_storage.delete(temp_path)
            print(f"[ERROR] Processing error: {e}")
            # Return a more user-friendly error message
            if "tensor" in str(e).lower():
                return JsonResponse({'error': 'Erreur de traitement audio. Veuillez réessayer.'}, status=500)
            else:
                return JsonResponse({'error': f'Erreur de traitement: {str(e)}'}, status=500)
            
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return JsonResponse({'error': f'Erreur générale: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_analyze_voice(request):
    """Stateless API: accepts audio or text and returns mood, message, theme, and music.
    Does not persist anything to the database.
    """
    try:
        transcription = ''
        wav_full_path = ''
        temp_paths = []
        if 'audio' in request.FILES:
            audio_file = request.FILES['audio']
            tmp_in = default_storage.save(f'temp/{audio_file.name}', ContentFile(audio_file.read()))
            temp_paths.append(tmp_in)
            full_in = default_storage.path(tmp_in)
            wav_buf = convert_to_wav(full_in)
            if wav_buf is None:
                return JsonResponse({'error': 'Failed to convert audio'}, status=400)
            tmp_wav = default_storage.save(f'temp/{os.path.basename(tmp_in)}.wav', ContentFile(wav_buf.getvalue()))
            temp_paths.append(tmp_wav)
            wav_full_path = default_storage.path(tmp_wav)
            try:
                asr = get_asr_pipeline()
                asr_result = asr(wav_full_path)
                transcription = asr_result.get('text', '')
            except Exception:
                transcription = ''
        else:
            # Accept either form field 'text' or raw body
            transcription = (request.POST.get('text') or request.body.decode('utf-8') or '').strip()

        # Text sentiment
        text_sentiment = 'NEUTRAL'
        text_sentiment_score = 0.0
        try:
            if transcription:
                text_sentiment, text_sentiment_score = heuristic_fr_sentiment(transcription)
                if USE_SENTIMENT_MODEL and text_sentiment == 'NEUTRAL':
                    sp = get_sentiment_pipeline()
                    sout = sp(transcription)[0]
                    text_sentiment, text_sentiment_score = normalize_sentiment_label(
                        sout.get('label', ''),
                        float(sout.get('score', 0.0))
                    )
        except Exception:
            pass

        # Audio emotion
        audio_emotion = 'neutral'
        audio_emotion_score = 0.0
        try:
            if wav_full_path:
                emo = analyze_audio_emotion(wav_full_path)
                # analyze_audio_emotion returns tuple in this file (emotion, score)
                if isinstance(emo, tuple):
                    audio_emotion, audio_emotion_score = emo
                else:
                    audio_emotion = emo.get('emotion', 'neutral')
                    audio_emotion_score = emo.get('score', 0.0)
        except Exception:
            pass

        final_mood = decide_mood(audio_emotion, text_sentiment)
        ai_message = generate_ai_message(final_mood)
        theme = EMOTION_TO_THEME.get(final_mood, EMOTION_TO_THEME['neutral'])
        music = get_youtube_music_recommendations(final_mood)

        return JsonResponse({
            'emotion': final_mood,
            'ai_message': ai_message,
            'theme': theme,
            'music': music,
            'transcription': transcription,
            'text_sentiment': text_sentiment,
            'text_sentiment_score': text_sentiment_score,
            'audio_emotion': audio_emotion,
            'audio_emotion_score': audio_emotion_score,
        })
    finally:
        for p in (temp_paths or []):
            try:
                default_storage.delete(p)
            except Exception:
                pass


@csrf_exempt
@require_http_methods(["POST"])
@login_required
def api_voice_agent_turn(request):
    """Accept a short voice turn, transcribe, generate empathetic reply, return JSON.
    Request form-data: file field 'audio' (webm/ogg/wav). No media persisted.
    """
    temp_paths = []
    try:
        transcription = ''
        final_mood = (request.POST.get('mood') or '').lower()  # optional hint from client

        # Start of conversation: AI speaks first, no audio required
        if (request.POST.get('start') or '') == '1':
            if not final_mood:
                final_mood = 'neutral'
            opening = generate_opening_prompt(final_mood)
            # Initialize simple dialog state in session
            request.session['agent_state'] = {
                'topic': 'support',  # support | breathing | end
                'step': 0,
                'mood': final_mood,
            }
            request.session.modified = True
            music = get_youtube_music_recommendations(final_mood)
            return JsonResponse({
                'success': True,
                'mood': final_mood,
                'user_text': '',
                'reply_text': opening,
                'music': music,
            })

        if 'audio' not in request.FILES:
            # Requête sans audio - initialiser le state si nécessaire
            state = request.session.get('agent_state')
            if not state:
                # Première requête sans start=1 - initialiser
                state = {
                    'topic': 'support',
                    'step': 0,
                    'mood': final_mood or 'neutral'
                }
                request.session['agent_state'] = state
                request.session.modified = True
            
            topic = state.get('topic', 'support')
            text = ''
            intent = 'unknown'
            
            # Appeler route_next pour obtenir la réponse
            try:
                reply, state_next, end_call_local, intent = route_next(state, text)
                state_next['mood'] = state.get('mood', final_mood)
                request.session['agent_state'] = state_next
                request.session.modified = True
                
                return JsonResponse({
                    'success': True,
                    'user_text': '',
                    'reply_text': reply,
                    'intent': intent,
                    'end_call': end_call_local,
                })
            except Exception as e:
                print(f"[Agent] Error in route_next: {e}")
                import traceback
                traceback.print_exc()
                return JsonResponse({'error': str(e)}, status=400)

        # Save temp input
        audio_file = request.FILES['audio']
        tmp_in = default_storage.save(f'temp/{audio_file.name}', ContentFile(audio_file.read()))
        temp_paths.append(tmp_in)
        full_in = default_storage.path(tmp_in)

        # Convert to wav
        wav_buf = convert_to_wav(full_in)
        if wav_buf is None:
            return JsonResponse({'error': 'Failed to convert audio'}, status=400)
        tmp_wav = default_storage.save(f'temp/{os.path.basename(tmp_in)}.wav', ContentFile(wav_buf.getvalue()))
        temp_paths.append(tmp_wav)
        wav_full_path = default_storage.path(tmp_wav)

        # STT
        try:
            asr = get_asr_pipeline()
            asr_result = asr(wav_full_path)
            transcription = asr_result.get('text', '')
        except Exception as e:
            print(f"[Agent] STT error: {e}")
            transcription = ''

        # If mood not provided, infer quickly from text only
        if not final_mood:
            text_sentiment = 'NEUTRAL'
            try:
                if transcription:
                    sp = get_sentiment_pipeline()
                    sout = sp(transcription)[0]
                    text_sentiment = sout.get('label', 'NEUTRAL')
            except Exception:
                pass
            final_mood = decide_mood('', text_sentiment)

        # Conversational state machine
        state = request.session.get('agent_state') or {'topic': 'support', 'step': 0, 'mood': final_mood}
        # Ensure mood in state
        if not state.get('mood'):
            state['mood'] = final_mood

        def route_next(state: dict, text: str) -> tuple[str, dict, bool, str]:
            intent = detect_user_intent_yes_no(text)
            t = (text or '').lower()
            topic = state.get('topic') or 'support'
            step = int(state.get('step') or 0)
            mood_s = (state.get('mood') or final_mood or 'neutral').lower()
            end_call_local = False
            
            # Debug logging
            print(f"[VoiceAgent] Route_next: text='{text}', intent='{intent}', topic='{topic}', step={step}")

            # Keyword router (no deep if chains; structured topic/steps)
            wants_breathing = any(k in t for k in ['respir', 'souffl', 'calmer', 'apaiser', 'relax'])
            wants_tips = any(k in t for k in ['conseil', 'astuce', 'propos', 'aide', 'idée', 'idee', 'donne', 'propose'])
            wants_music = any(k in t for k in ['musique', 'music', 'chanson', 'écouter', 'ecouter', 'playlist', 'lofi'])

            # Handle music playback
            if topic == 'music':
                # Let LLM handle the conversation during music playback
                conversation_history = state.get('history', [])
                reply = generate_llm_reply(text, mood_s, conversation_history)
                return reply, state, False, intent

            # Support mode - mostly handled by LLM
            if topic == 'support':
                
                # Check if user explicitly wants music or breathing BEFORE asking LLM
                text_lower = text.lower()
                if 'musique' in text_lower or 'music' in text_lower:
                    # User wants music
                    vids = get_youtube_music_recommendations(mood_s)
                    if vids:
                        state_next = {'topic': 'music', 'step': 0, 'mood': mood_s, 'vids': vids}
                        return "Extrait musical.", state_next, False, intent
                
                if 'respir' in text_lower or 'exercice' in text_lower:
                    # User wants breathing
                    topic = 'breathing'
                    step = 0
                    state['topic'] = topic
                    state['step'] = step
                    return reply, state, False, intent
                
                # Use LLM for ALL other replies
                conversation_history = state.get('history', [])
                reply = generate_llm_reply(text, mood_s, conversation_history)
                
                # Update conversation history
                new_history = conversation_history + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply}
                ]
                state['history'] = new_history[-6:]  # Keep last 6 messages
                
                return reply, state, False, intent

            # Breathing protocol (box breathing like 4-4-6, then reflect)
            if topic == 'breathing':
                introduction = "D'accord. On commence une respiration simple."
                guide = [
                    introduction,  # step 0: Introduction
                    "Inspire 4 secondes.",  # step 1: Première instruction
                    "Bloque 4 secondes.",  # step 2
                    "Expire doucement 6 secondes.",  # step 3
                    "Très bien. On refait une fois si tu veux. Dis 'oui' pour refaire, sinon 'non'.",  # step 4
                ]
                
                # Si step est 0 (début), donner l'introduction et passer à step 1
                if step == 0:
                    reply = guide[0]
                    state['topic'] = 'breathing'
                    state['step'] = 1
                    return reply, state, False, intent
                
                # Si step est 1, 2, ou 3, donner automatiquement l'étape suivante sans attendre
                elif step < len(guide):
                    reply = guide[step]
                    state['topic'] = 'breathing'
                    state['step'] = step + 1
                    return reply, state, False, intent
                
                # Si on est à la fin (step == len(guide)), demander si on refait
                else:
                    state['auto_continue'] = False
                    if intent == 'yes':
                        state['step'] = 0
                        reply = guide[0]
                        state['topic'] = topic
                    elif intent == 'no':
                        reply = (
                            "Très bien. Est-ce que tu veux écouter de la musique apaisante, "
                            "ou as-tu besoin d'autre chose ? Dis-moi ce que tu veux."
                        )
                        topic = 'support'
                        state['topic'] = topic
                        state['step'] = 0
                        state['last_message'] = reply
                    else:
                        reply = "Souhaites-tu refaire un cycle de respiration ?"
                
                return reply, state, False, intent


            # Fallback - use LLM for any remaining cases
            conversation_history = state.get('history', [])
            reply = generate_llm_reply(text, mood_s, conversation_history)
            new_history = conversation_history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply}
            ]
            state['history'] = new_history[-6:]
            return reply, state, False, intent

        if not transcription:
            reply_text = "Je n'ai pas bien entendu. Peux-tu répéter en quelques mots ?"
            end_call = False
            intent = 'unknown'
        else:
            reply_text, new_state, end_call, intent = route_next(state, transcription)
            # Stocker le dernier message pour le contexte
            new_state['last_message'] = reply_text
            request.session['agent_state'] = new_state
            request.session.modified = True

        # Optional preview payload if in music topic
        preview = None
        ns = request.session.get('agent_state') or {}
        if ns.get('topic') == 'music':
            vids = ns.get('vids') or get_youtube_music_recommendations(final_mood)
            step = int(ns.get('step') or 0)
            if vids and step < len(vids):
                vid = vids[step]
                embed = (vid.get('embedUrl') or '')
                if embed:
                    joiner = '&' if '?' in embed else '?'
                    embed = f"{embed}{joiner}autoplay=1&start=0&end=15&mute=0&modestbranding=1"
                    preview = {
                        'title': vid.get('title') or 'Musique apaisante',
                        'embedUrl': embed,
                    }

        music = get_youtube_music_recommendations(final_mood)

        return JsonResponse({
            'success': True,
            'mood': final_mood,
            'user_text': transcription,
            'reply_text': reply_text,
            'music': music,
            'end_call': end_call,
            'intent': intent,
            'preview': preview,
        })
    except Exception as e:
        print(f"[Agent] turn error: {e}")
        return JsonResponse({'error': 'Agent error'}, status=500)
    finally:
        for p in (temp_paths or []):
            try:
                default_storage.delete(p)
            except Exception:
                pass
