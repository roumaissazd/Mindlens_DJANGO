from deepface import DeepFace
import numpy as np
import cv2
import os
from django.conf import settings
from django.contrib.auth.models import User
import logging

logger = logging.getLogger(__name__)

def detect_faces(image_path):
    """Détecte tous les visages dans une image"""
    try:
        # Utiliser OpenCV pour la détection initiale des visages
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        logger.info(f"Nombre de visages détectés : {len(faces)}")
        
        face_regions = []
        for (x, y, w, h) in faces:
            # Ajouter une marge autour du visage
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            face_regions.append({
                'region': image[y1:y2, x1:x2],
                'location': {
                    'left': int(x),
                    'top': int(y),
                    'right': int(x + w),
                    'bottom': int(y + h)
                }
            })
        
        return face_regions
    except Exception as e:
        logger.error(f"Erreur lors de la détection des visages : {str(e)}")
        return []

def compare_with_users(image_path):
    """Compare tous les visages d'une image avec les photos de profil des utilisateurs"""
    logger.info(f"Début de la comparaison pour l'image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"L'image source n'existe pas: {image_path}")
        return []

    # Détecter tous les visages dans l'image
    face_regions = detect_faces(image_path)
    logger.info(f"Nombre de régions de visages détectées : {len(face_regions)}")

    results = []
    users = User.objects.filter(profile__photo_user__isnull=False)
    
    try:
        # Pour chaque visage détecté
        for face_region in face_regions:
            face_matches = []
            
            # Comparer avec chaque utilisateur
            for user in users:
                profile_photo_path = os.path.join(settings.MEDIA_ROOT, str(user.profile.photo_user))
                
                if not os.path.exists(profile_photo_path):
                    continue
                    
                try:
                    # Sauvegarder temporairement la région du visage
                    temp_face_path = os.path.join(settings.MEDIA_ROOT, 'temp', f'face_{hash(str(face_region))}.jpg')
                    os.makedirs(os.path.dirname(temp_face_path), exist_ok=True)
                    cv2.imwrite(temp_face_path, face_region['region'])
                    
                    # Comparer les visages
                    result = DeepFace.verify(
                        img1_path=profile_photo_path,
                        img2_path=temp_face_path,
                        model_name="VGG-Face",
                        detector_backend="opencv",
                        distance_metric="cosine",
                        enforce_detection=False,
                        align=True
                    )
                    
                    # Supprimer le fichier temporaire
                    os.remove(temp_face_path)
                    
                    if result.get("verified", False):
                        confidence = (1 - result.get("distance", 0)) * 100
                        face_matches.append({
                            'username': user.username,
                            'confidence': confidence,
                            'location': face_region['location']
                        })
                        logger.info(f"Match trouvé pour {user.username} avec {confidence:.2f}% de confiance")
                
                except Exception as e:
                    logger.error(f"Erreur lors de la comparaison avec {user.username}: {str(e)}")
                    continue
            
            # Ajouter le meilleur match pour ce visage
            if face_matches:
                best_match = max(face_matches, key=lambda x: x['confidence'])
                results.append(best_match)
    
    except Exception as e:
        logger.error(f"Erreur globale lors de la comparaison: {str(e)}")
    
    logger.info(f"Comparaison terminée. {len(results)} matches trouvés.")
    return results