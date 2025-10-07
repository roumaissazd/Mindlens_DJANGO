import os
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
import os
import requests
from .models import VoiceJournal
import time

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
    """Combine audio emotion + text sentiment to a final mood label.
    Falls back to audio_emotion if available, else sentiment mapping.
    """
    sentiment_map = {
        'POSITIVE': 'happy',
        'NEGATIVE': 'sad',
        'NEUTRAL':  'neutral'
    }
    normalized_audio = (audio_emotion or '').lower().strip()
    if normalized_audio in {'happy', 'sad', 'calm', 'stressed', 'neutral'}:
        return normalized_audio
    return sentiment_map.get((text_sentiment or '').upper(), 'neutral')


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
    
    # Always create a fresh pipeline to avoid tensor size conflicts
    try:
        # Uses Hugging Face Whisper model locally
        # Small/base models are faster and lighter
        _asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            device="cpu",
            chunk_length_s=30,
            return_timestamps=False,
        )
        print("[ASR] Fresh Whisper pipeline created")
    except Exception as e:
        print(f"[ASR] Error creating pipeline: {e}")
        # Reset global variable on error
        _asr_pipeline = None
        raise e
    
    return _asr_pipeline


def get_sentiment_pipeline():
    """Get or initialize sentiment analysis pipeline"""
    if not ML_LIBRARIES_AVAILABLE:
        raise ImportError("ML libraries not available")
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis")
    return _sentiment_pipeline


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
        # Read audio file
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
    """List all voice journal entries for the current user"""
    voice_entries = VoiceJournal.objects.filter(user=request.user)
    
    # Pagination
    paginator = Paginator(voice_entries, 10)  # 10 entries per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'voice_entries': page_obj,
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
            try:
                asr = get_asr_pipeline()
                asr_result = asr(wav_full_path)
                transcription = asr_result.get("text", "")
                print(f"[ASR] Transcription successful: '{transcription}'")
            except Exception as e:
                print(f"[ASR] Error during transcription: {e}")
                # Try to reset and retry once
                try:
                    global _asr_pipeline
                    _asr_pipeline = None  # Force reset
                    asr = get_asr_pipeline()
                    asr_result = asr(wav_full_path)
                    transcription = asr_result.get("text", "")
                    print(f"[ASR] Retry successful: '{transcription}'")
                except Exception as retry_e:
                    print(f"[ASR] Retry failed: {retry_e}")
                    transcription = "Erreur de transcription - veuillez réessayer"
            t_asr1 = time.time()
            
            # Analyze text sentiment
            t_sent0 = time.time()
            sentiment_pipeline = get_sentiment_pipeline()
            sentiment_result = sentiment_pipeline(transcription)[0]
            text_sentiment = sentiment_result['label']
            text_sentiment_score = sentiment_result['score']
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
                sp = get_sentiment_pipeline()
                sout = sp(transcription)[0]
                text_sentiment = sout.get('label', 'NEUTRAL')
                text_sentiment_score = float(sout.get('score', 0.0))
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
