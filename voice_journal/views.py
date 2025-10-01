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
from .models import VoiceJournal
import time

# Import ML libraries with error handling
ML_LIBRARIES_AVAILABLE = True
ML_MISSING_MODULES = []
try:
    import io
    import torch
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"torch ({e})")
try:
    from transformers import pipeline
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"transformers ({e})")
try:
    import librosa
    import numpy as np
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"librosa/numpy ({e})")
try:
    from pydub import AudioSegment
except Exception as e:
    ML_LIBRARIES_AVAILABLE = False
    ML_MISSING_MODULES.append(f"pydub ({e})")
if not ML_LIBRARIES_AVAILABLE:
    print(f"Warning: ML libraries not available: {', '.join(ML_MISSING_MODULES)}")


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
