"""
AI utilities for note analysis using Hugging Face transformers.
Includes sentiment analysis, category classification, and tag generation.
"""
import base64
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import logging
from io import BytesIO
from gtts import gTTS
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import torch

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0
# Global pipelines (loaded once for performance)
_sentiment_pipeline = None
_zero_shot_pipeline = None


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




# Cache global pour éviter de recharger le modèle à chaque appel
def get_summarizer():
    """
    Initialise le pipeline de résumé IA uniquement à la demande.
    Cela évite le téléchargement du modèle au démarrage de Django.
    """
    return pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary_from_notes(notes_contents, chunk_size=1000, max_summary_length=150):
    """
    Génère un résumé à partir d'une liste de textes (notes).
    
    notes_contents : list de str
    chunk_size : nombre de caractères par chunk
    max_summary_length : longueur max pour chaque chunk résumé
    """
    if not notes_contents:
        return "Aucune note à résumer."

    summarizer = get_summarizer()
    full_text = " ".join(notes_contents)

    # Découper le texte en chunks
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    final_summary = ""
    
    for chunk in chunks:
        try:
            result = summarizer(
                chunk, 
                max_length=max_summary_length, 
                min_length=50, 
                do_sample=False
            )
            final_summary += result[0]['summary_text'] + " "
        except Exception as e:
            # En cas d'erreur sur un chunk, ignorer et continuer
            final_summary += ""
            print(f"Erreur IA chunk: {e}")

    return final_summary.strip()

# --------------------------------------------------------------
#  NEW: language-aware TTS → base64 MP3
# --------------------------------------------------------------
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# make results deterministic (optional but nice)
DetectorFactory.seed = 0




def _detect_language(text: str) -> str:
    """
    Detect language code (ISO-639-1) of the first non-empty line.
    Returns 'fr' as safe default.
    """
    # gTTS only needs the first part of the text for detection
    sample = text.strip()[:500]
    if not sample:
        return "fr"

    try:
        code = detect(sample)
        # gTTS supports only a subset of codes → map the most common ones
        mapping = {
            "fr": "fr", "en": "en", "es": "es", "de": "de", "it": "it",
            "pt": "pt", "nl": "nl", "ru": "ru", "zh-cn": "zh-CN",
            "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi",
        }
        return mapping.get(code, "fr")
    except LangDetectException:
        return "fr"


def text_to_speech_base64(text: str, lang: str | None = None) -> str:
    """
    Convert *text* → MP3 → base64 string.
    If *lang* is None → auto-detect language.
    """
    if not text.strip():
        return ""

    # --------------------------------------------------
    # 1. Choose language
    # --------------------------------------------------
    if lang is None:
        lang = _detect_language(text)          # ← auto-detect
    else:
        # keep user-provided code, but normalise a bit
        lang = lang.lower().split("-")[0]

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        logger.info(f"TTS generated – lang:{lang}")
        return b64
    except Exception as e:
        logger.error(f"TTS error (lang={lang}): {e}")
        return ""
    



_title_generator = None

def get_title_generator():
    """Générateur de titre français – MODÈLE ROBUSTE (T5)"""
    global _title_generator, _title_tokenizer
    if _title_generator is None:
        logger.info("Chargement du générateur de titre français (T5)...")
        try:
            ckpt = "plguillou/t5-base-fr-sum-cnndm"  # Modèle français SANS BUG
            _title_tokenizer = T5Tokenizer.from_pretrained(ckpt)
            _title_generator = T5ForConditionalGeneration.from_pretrained(ckpt)
            device = 0 if torch.cuda.is_available() else 'cpu'
            _title_generator.to(device)
            logger.info("Générateur de titre français (T5) chargé !")
        except Exception as e:
            logger.error(f"Échec chargement titre: {e}")
            _title_generator = None
            _title_tokenizer = None
    return _title_generator, _title_tokenizer


def generate_summary_title(
    notes_contents: list[str],
    period: str,
    category: str | None = None,
    max_words: int = 10
) -> str:
    full_text = " ".join(notes_contents)[:1500]

    # Sentiment
    sentiment = analyze_sentiment(full_text)
    sentiment_word = ""
    if sentiment:
        mood = suggest_mood(sentiment["label"])
        sentiment_word = {"joyeux": "positif", "triste": "négatif", "neutre": ""}.get(mood, "")

    # Category fallback
    if not category:
        cat_res = classify_category(full_text)
        category = cat_res["category"] if cat_res else None

    # Tags
    tags = generate_tags(full_text, max_tags=3)
    key_phrase = " ".join(tags[:2]).strip()

    # AI generation
    prompt = (
        f"Résume en une phrase courte et accrocheuse les notes suivantes : "
        f"{' '.join(notes_contents)[:800]}. "
        f"Thème principal : {key_phrase or 'divers'}. "
        f"Sentiment : {sentiment_word or 'neutre'}."
    )

    generator = get_title_generator()
    if generator:
        try:
            result = generator(prompt, num_return_sequences=1)[0]["generated_text"]
            title = result.strip().split("\n")[0].split(".")[0].split("!")[0].split("?")[0].strip()
            words = title.split()
            if len(words) > max_words:
                title = " ".join(words[:max_words]) + "…"
            if title:
                return title
        except Exception as e:
            logger.warning(f"AI title failed: {e}")

    # Fallback
    parts = []
    if sentiment_word:
        parts.append(sentiment_word.capitalize())
    if key_phrase:
        parts.append(key_phrase.capitalize())
    if category:
        parts.append(category.capitalize())
    if period == "week":
        parts.append("de la semaine")
    elif period == "month":
        parts.append("du mois")

    fallback = " ".join(p for p in parts if p) or "Résumé"
    return fallback