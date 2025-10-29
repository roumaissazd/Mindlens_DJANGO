"""
AI utilities for note analysis using Hugging Face transformers.
Includes sentiment analysis, category classification, and tag generation.
"""

from transformers import pipeline
import logging
from .models import Reminder
from django.utils import timezone
from datetime import timedelta
logger = logging.getLogger(__name__)

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