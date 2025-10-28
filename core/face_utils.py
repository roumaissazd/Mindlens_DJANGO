from deepface import DeepFace
import numpy as np
import cv2
import os
from django.conf import settings
from django.contrib.auth.models import User
import logging

logger = logging.getLogger(__name__)

def verify_image_exists(image_path):
    """Vérifie si l'image existe et log les informations"""
    exists = os.path.exists(image_path)
    logger.info(f"Vérification image {image_path}: {'Existe' if exists else 'N\'existe pas'}")
    if exists:
        try:
            size = os.path.getsize(image_path)
            logger.info(f"Taille de l'image: {size} bytes")
        except:
            logger.error(f"Impossible de lire la taille de {image_path}")
    return exists

def compare_with_users(image_path):
    """Compare les visages d'une image avec les photos de profil des utilisateurs"""
    logger.info(f"Début de la comparaison pour l'image: {image_path}")
    
    if not verify_image_exists(image_path):
        logger.error(f"L'image source n'existe pas: {image_path}")
        return []

    results = []
    users = User.objects.filter(profile__photo_user__isnull=False)
    logger.info(f"Nombre d'utilisateurs avec photo: {users.count()}")
    
    try:
        for user in users:
            profile_photo_path = os.path.join(settings.MEDIA_ROOT, str(user.profile.photo_user))
            logger.info(f"Traitement de l'utilisateur {user.username}")
            logger.info(f"Chemin de la photo de profil: {profile_photo_path}")
            
            if not verify_image_exists(profile_photo_path):
                continue
                
            try:
                # Utiliser directement les chemins de fichiers
                result = DeepFace.verify(
                    img1_path=profile_photo_path,
                    img2_path=image_path,
                    model_name="VGG-Face",
                    detector_backend="opencv",  # Plus stable que retinaface
                    distance_metric="cosine",
                    enforce_detection=False,
                    align=True
                )
                
                logger.info(f"Résultat de comparaison pour {user.username}: {result}")
                
                if result.get("verified", False):
                    confidence = (1 - result.get("distance", 0)) * 100
                    
                    # Détecter la position du visage avec OpenCV
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        location = {
                            'left': int(x),
                            'top': int(y),
                            'right': int(x + w),
                            'bottom': int(y + h)
                        }
                    else:
                        # Position par défaut si pas de visage détecté
                        location = {
                            'left': 0,
                            'top': 0,
                            'right': img.shape[1],
                            'bottom': img.shape[0]
                        }
                    
                    results.append({
                        'username': user.username,
                        'location': location,
                        'confidence': confidence
                    })
                    logger.info(f"Match trouvé pour {user.username} avec {confidence:.2f}% de confiance")
            
            except Exception as e:
                logger.error(f"Erreur lors de la comparaison avec {user.username}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Erreur globale lors de la comparaison: {str(e)}")
    
    logger.info(f"Comparaison terminée. {len(results)} matches trouvés.")
    return results