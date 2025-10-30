"""
AI utilities for note analysis using Hugging Face transformers.
Includes sentiment analysis, category classification, and tag generation.
"""
import base64
from transformers import pipeline
import logging
from .models import Reminder
from django.utils import timezone
from datetime import timedelta
from io import BytesIO
from gtts import gTTS
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from typing import List, Dict, Any
import torch
import requests
from .models import Note
from django.db.models.signals import post_save
from django.dispatch import receiver
import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0
# Global pipelines (loaded once for performance)
_sentiment_pipeline = None
_zero_shot_pipeline = None
_summarizer = None
_title_generator = None
_advice_generator = None

MYMEMORY_URL = "https://api.mymemory.translated.net/get"

def get_sentiment_pipeline():
    """Get or create sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            # Using multilingual sentiment model
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            logger.info("Sentiment analysis pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment pipeline: {e}")
            _sentiment_pipeline = None
    return _sentiment_pipeline


def get_zero_shot_pipeline():
    """Get or create zero-shot classification pipeline."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        try:
            # Using multilingual zero-shot model
            _zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("Zero-shot classification pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading zero-shot pipeline: {e}")
            _zero_shot_pipeline = None
    return _zero_shot_pipeline


def analyze_sentiment(text):
    """
    Analyze sentiment of text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: {'label': str, 'score': float} or None if error
    """
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        pipeline = get_sentiment_pipeline()
        if pipeline is None:
            return None
        
        # Truncate text if too long (max 512 tokens)
        text_truncated = text[:2000]
        
        result = pipeline(text_truncated)[0]
        
        # Convert star rating to sentiment label
        star_to_sentiment = {
            '1 star': 'très négatif',
            '2 stars': 'négatif',
            '3 stars': 'neutre',
            '4 stars': 'positif',
            '5 stars': 'très positif',
        }
        
        sentiment_label = star_to_sentiment.get(result['label'], 'neutre')
        
        return {
            'label': sentiment_label,
            'score': result['score']
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return None


def classify_category(text):
    """
    Classify text into predefined categories using zero-shot classification.
    
    Args:
        text (str): Text to classify
        
    Returns:
        dict: {'category': str, 'score': float, 'all_scores': dict} or None if error
    """
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        pipeline = get_zero_shot_pipeline()
        if pipeline is None:
            return None
        
        # Categories to classify into
        categories = [
            'famille',
            'travail',
            'voyage',
            'santé',
            'amour',
            'loisirs',
            'réflexion',
            'autre'
        ]
        
        # Truncate text if too long
        text_truncated = text[:1000]
        
        result = pipeline(
            text_truncated,
            candidate_labels=categories,
            multi_label=False
        )
        
        # Get top category
        top_category = result['labels'][0]
        top_score = result['scores'][0]
        
        # Create scores dict
        all_scores = {
            label: score 
            for label, score in zip(result['labels'], result['scores'])
        }
        
        return {
            'category': top_category,
            'score': top_score,
            'all_scores': all_scores
        }
    except Exception as e:
        logger.error(f"Error classifying category: {e}")
        return None


def generate_tags(text, max_tags=5):
    """
    Generate relevant tags from text using keyword extraction.
    
    Args:
        text (str): Text to extract tags from
        max_tags (int): Maximum number of tags to generate
        
    Returns:
        list: List of tag strings
    """
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        # Simple keyword extraction based on word frequency
        # Remove common French stop words
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou',
            'mais', 'donc', 'car', 'ni', 'que', 'qui', 'quoi', 'dont',
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
            'ce', 'cet', 'cette', 'ces', 'à', 'au', 'aux', 'dans', 'par',
            'pour', 'sur', 'avec', 'sans', 'sous', 'vers', 'chez',
            'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir',
            'pouvoir', 'vouloir', 'devoir', 'falloir', 'prendre', 'venir',
            'très', 'plus', 'moins', 'bien', 'mal', 'beaucoup', 'peu',
            'trop', 'assez', 'aussi', 'encore', 'déjà', 'toujours', 'jamais',
            'aujourd', 'hui', 'demain', 'hier', 'maintenant', 'puis', 'alors',
        }
        
        # Clean and tokenize
        import re
        words = re.findall(r'\b[a-zàâäéèêëïîôùûüÿæœç]{3,}\b', text.lower())
        
        # Filter stop words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and get top tags
        # Changed: Accept words that appear at least once (removed freq > 1 condition)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        tags = [word for word, freq in sorted_words[:max_tags]]

        return tags
    except Exception as e:
        logger.error(f"Error generating tags: {e}")
        return []


def suggest_mood(sentiment_label):
    """
    Suggest a mood based on sentiment analysis.

    Args:
        sentiment_label (str): Sentiment label from analyze_sentiment

    Returns:
        str: Suggested mood (joyeux, triste, neutre, etc.)
    """
    mood_mapping = {
        'Très positif': 'joyeux',
        'Positif': 'joyeux',
        'Neutre': 'neutre',
        'Négatif': 'triste',
        'Très négatif': 'triste',
    }
    return mood_mapping.get(sentiment_label, 'neutre')


def analyze_note(text):
    """
    Perform complete AI analysis on a note.

    Args:
        text (str): Note content to analyze

    Returns:
        dict: {
            'sentiment': dict or None,
            'category': dict or None,
            'tags': list,
            'suggested_mood': str or None
        }
    """
    sentiment = analyze_sentiment(text)
    suggested_mood = None

    if sentiment and sentiment.get('label'):
        suggested_mood = suggest_mood(sentiment['label'])

    return {
        'sentiment': sentiment,
        'category': classify_category(text),
        'tags': generate_tags(text),
        'suggested_mood': suggested_mood
    }

def create_reminder_from_analysis(note, analysis):
  
    try:
        sentiment = analysis.get('sentiment', {})
        category = analysis.get('category', {})
        tags = analysis.get('tags', [])
        mood = analysis.get('suggested_mood', '')

        priority = 'basse'
        delay_hours = 24

        # 1. Sentiment négatif
        sentiment_label = sentiment.get('label', '').lower()
        if 'négatif' in sentiment_label:
            priority = 'haute'
            delay_hours = 1

        # 2. Catégorie santé ou travail
        cat = category.get('category', '').lower()
        if cat in ['santé', 'travail']:
            priority = 'moyenne'
            delay_hours = 6

        # 3. Tags urgents
        urgent_keywords = ['médecin', 'rdv', 'urgent', 'deadline', 'oublié', 'demain', 'aujourd’hui', 'maintenant']
        text_lower = note.content.lower()
        if any(kw in text_lower for kw in urgent_keywords):
            priority = 'haute'
            delay_hours = 0.001  # 1 minute pour test

        # Pas d'urgence → pas de rappel
        if priority == 'basse':
            logger.info("Aucun rappel créé : pas d'urgence détectée")
            return None

        # Message court
        preview = note.content[:50].strip()
        if len(note.content) > 50:
            preview += '...'
        message = f"{cat.capitalize() if cat else 'Note'} : {preview}"

        # Créer ou mettre à jour
        reminder, created = Reminder.objects.update_or_create(
            note=note,
            user=note.user,
            defaults={
                'message': message,
                'priority': priority,
                'trigger_at': timezone.now() + timedelta(hours=delay_hours),
                'is_read': False
            }
        )
        action = 'créé' if created else 'mis à jour'
        logger.info(f"Reminder {action} : {reminder} (priorité: {priority}, dans {delay_hours}h)")
        return reminder

    except Exception as e:
        logger.error(f"Erreur reminder: {e}")
        return None


# =========================================================
# 6. SUMMARIZATION
# =========================================================
def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            device = 0 if torch.cuda.is_available() else -1  # GPU si dispo
            # Charger explicitement le tokenizer "slow" (SentencePiece)
            tok = AutoTokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm", use_fast=False)
            mdl = AutoModelForSeq2SeqLM.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
            _summarizer = pipeline(
                "summarization",
                model=mdl,
                tokenizer=tok,
                device=device
            )
            logger.info("Summarizer chargé avec succès !")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du summarizer : {e}", exc_info=True)
            _summarizer = None
    return _summarizer


def generate_summary_from_notes(
    notes_contents: List[str],
    chunk_size: int = 1000,
    max_summary_length: int = 150
) -> str:
    if not notes_contents:
        return "Aucune note à résumer."

    summarizer = get_summarizer()
    if not summarizer:
        return "Résumé indisponible (modèle non chargé)."

    full_text = " ".join(notes_contents)
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    final_summary = []
    for chunk in chunks:
        try:
            result = summarizer(
                chunk,
                max_length=max_summary_length,
                min_length=50,
                do_sample=False
            )
            final_summary.append(result[0]['summary_text'])
        except Exception as e:
            logger.warning(f"Summary chunk failed: {e}")

    return " ".join(final_summary).strip()


# =========================================================
# 7. TEXT-TO-SPEECH (base64 MP3)
# =========================================================
def _detect_language(text: str) -> str:
    sample = text.strip()[:500]
    if not sample:
        return "fr"

    try:
        code = detect(sample)
        mapping = {
            "fr": "fr", "en": "en", "es": "es", "de": "de", "it": "it",
            "pt": "pt", "nl": "nl", "ru": "ru", "zh-cn": "zh-CN",
            "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi",
        }
        return mapping.get(code, "fr")
    except LangDetectException:
        return "fr"


def text_to_speech_base64(text: str, lang: str | None = None) -> str:
    if not text.strip():
        return ""

    lang = lang or _detect_language(text)
    lang = lang.lower().split("-")[0]

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    except Exception as e:
        logger.error(f"TTS error (lang={lang}): {e}")
        return ""


def translate_note_mymemory(note, target_language='en'):
    """
    Traduit une note avec MyMemory (gratuit, sans clé).
    """
    if not note.content or len(note.content.strip()) == 0:
        return None

    try:
        # Langue source = français
        params = {
            'q': note.content,
            'langpair': f'fr|{target_language}'  # fr→en, fr→es, etc.
        }

        response = requests.get(MYMEMORY_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        translated = data['responseData']['translatedText'].strip()
        
        # Évite les traductions vides ou "NO QUERY"
        if translated in ["", "NO QUERY SPECIFIED", "ERROR"]:
            return None
        
        # Sauvegarde
        note.translated_content = translated
        note.translated_language = target_language
        note.save()
        
        logger.info(f"Note {note.id} traduite via MyMemory en {target_language}")
        return translated

    except Exception as e:
        logger.error(f"Erreur MyMemory : {e}")
        return None


def get_title_generator():
    global _title_generator
    if _title_generator is None:
        try:
            # MODÈLE SPÉCIALISÉ EN TITRES (anglais → français OK)
            _title_generator = pipeline(
                "text2text-generation",
                model="Michau/t5-base-en-generate-headline",
                framework="pt"
            )
            logger.info("Title generator (T5 headline) loaded")
        except Exception as e:
            logger.error(f"Erreur titre IA : {e}")
            _title_generator = None
    return _title_generator

def generate_title(text: str) -> str:
    """
    Génère un titre court, clair et pertinent SANS modèle IA lourd.
    Basé sur : mots fréquents, verbes d'action, lieux, dates, émotions.
    """
    if not text or len(text.strip()) < 15:
        return "Ma note"

    try:
        # Nettoyer le texte
        text = text.strip()
        text_lower = text.lower()

        # 1. Détecter les phrases courtes (idéal pour titre)
        sentences = re.split(r'[.!?]\s*', text)
        short_sentences = [s.strip() for s in sentences if 15 < len(s) < 80]
        if short_sentences:
            # Prendre la première phrase courte
            candidate = short_sentences[0]
            if len(candidate.split()) <= 12:
                return _clean_title(candidate)

        # 2. Mots-clés prioritaires (verbes, lieux, émotions, dates)
        priority_keywords = [
            # Verbes d'action
            'couru', 'mangé', 'vu', 'rencontré', 'appelé', 'pensé', 'écrit', 'lu', 'fait',
            'voyagé', 'travaillé', 'dormi', 'pleuré', 'ri', 'aimé', 'détesté',
            # Lieux
            'parc', 'plage', 'maison', 'travail', 'école', 'médecin', 'hôpital', 'restaurant',
            'ville', 'montagne', 'forêt', 'jardin',
            # Temps
            'aujourd’hui', 'hier', 'demain', 'matin', 'soir', 'nuit',
            # Émotions
            'heureux', 'triste', 'fatigué', 'excité', 'calme', 'en colère', 'perdu', 'soulagé'
        ]

        found = []
        for word in priority_keywords:
            if word in text_lower:
                # Trouver le mot dans le texte original (conserve la casse)
                match = re.search(rf'\b{re.escape(word)}\b', text_lower)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    # Extraire une phrase courte autour
                    phrase = re.search(r'[^.!?]*\b' + re.escape(word) + r'\b[^.!?]*[.!?]?', context)
                    if phrase:
                        clean = phrase.group(0).strip()
                        if 10 < len(clean) < 70 and len(clean.split()) <= 10:
                            found.append(clean)
        
        if found:
            return _clean_title(found[0])

        # 3. Extraire les 2-3 premiers mots significatifs
        words = re.findall(r'\b[a-zàâäéèêëïîôùûüÿçœæ]{3,}\b', text_lower)
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais',
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'mon', 'ma',
            'ton', 'ta', 'son', 'sa', 'ces', 'ce', 'cette', 'cet', 'dans', 'sur',
            'pour', 'avec', 'sans', 'par', 'chez', 'très', 'plus', 'aussi', 'alors'
        }
        filtered = [w for w in words if w not in stop_words][:6]
        
        if len(filtered) >= 2:
            # Reconstruire avec majuscules
            title = ' '.join([_capitalize_word(w, text) for w in filtered[:3]])
            if len(title) < 50:
                return title

        # 4. Fallback : première phrase courte
        first_sentence = text.split('.')[0].split('!')[0].split('?')[0].strip()
        if 15 < len(first_sentence) < 80:
            return _clean_title(first_sentence)

        return "Note du jour"

    except Exception as e:
        logger.error(f"Erreur génération titre : {e}")
        return "Note importante"


def _capitalize_word(word: str, original_text: str) -> str:
    """Capitalise le mot comme dans le texte original"""
    match = re.search(rf'\b{re.escape(word)}\b', original_text, re.IGNORECASE)
    if match:
        return match.group(0)
    return word.capitalize()


def _clean_title(title: str) -> str:
    """Nettoie et formate le titre"""
    title = title.strip('.,!? ')
    title = re.sub(r'\s+', ' ', title)
    if len(title) > 70:
        title = title[:67] + '...'
    return title.capitalize()



@receiver(post_save, sender=Note)
def auto_generate_note_title(sender, instance, created, **kwargs):
    """
    Génère automatiquement un titre IA si la note est nouvelle et sans titre.
    """
    if created and not instance.title.strip():
        title = generate_title(instance.content)
        instance.title = title
        instance.save(update_fields=['title'])
        logger.info(f"Titre IA généré pour note {instance.id} : '{title}'")


def get_advice_generator():
    global _advice_generator
    if _advice_generator is None:
        _advice_generator = pipeline(
            "text2text-generation",
            model="moussaKam/barthez-orangesum-title",
            max_length=100
        )
    return _advice_generator

def generate_smart_advice(text: str) -> str:
    """
    IA qui donne un conseil personnalisé intelligent.
    """
    if not text or len(text) < 10:
        return "Prends soin de toi."

    # Détecte des mots clés
    text_lower = text.lower()
    conseils = []

    if any(word in text_lower for word in ['fatigué', 'fatigue', 'dormir', 'sommeil']):
        conseils.append("Bois de l’eau, dors tôt, évite les écrans.")
    if any(word in text_lower for word in ['médecin', 'rdv', 'rendez-vous']):
        conseils.append("Programme ton appel médecin demain matin.")
    if any(word in text_lower for word in ['triste', 'perdu', 'reflexion']):
        conseils.append("Écris tes pensées, respire, parle à quelqu’un.")
    if any(word in text_lower for word in ['parc', 'couru', 'sport']):
        conseils.append("Super ! Continue, ton corps te remercie.")
    if any(word in text_lower for word in ['deadline', 'travail', 'réunion']):
        conseils.append("Fais une pause de 5 min toutes les heures.")

    # Si pas de règle → IA génère
    if not conseils:
        try:
            generator = get_advice_generator()
            prompt = f"Donne un conseil court et bienveillant pour : {text[:200]}"
            result = generator(prompt, max_length=60)[0]['generated_text']
            return result.strip()
        except:
            return "Écoute ton corps, il sait ce dont tu as besoin."

    # Sinon, combine les conseils
    return " ".join(conseils[:2])