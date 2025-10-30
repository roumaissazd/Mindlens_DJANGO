import os
import logging
from typing import Optional, Any, List, Dict
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("DeepFace not available, face recognition will be limited")

from django.conf import settings
from django.contrib.auth.models import User

from .models import Photo, PhotoAlbum, Profile

logger = logging.getLogger(__name__)

# Preferred detector backends to try (order = try first -> fallbacks)
ENV_BACKEND = os.environ.get("DETECTOR_BACKEND", "").strip().lower()
if DEEPFACE_AVAILABLE:
    PREFERRED_BACKENDS = ([ENV_BACKEND] if ENV_BACKEND else []) + ['opencv', 'mtcnn', 'ssd', 'dlib']
else:
    PREFERRED_BACKENDS = ['opencv']  # Fallback to OpenCV only
# Ensure unique order
PREFERRED_BACKENDS = [b for i, b in enumerate(PREFERRED_BACKENDS) if b and b not in PREFERRED_BACKENDS[:i]]

# Configuration
MODEL_NAME = "Facenet"  # or "VGG-Face", "OpenFace", "DeepFace", "ArcFace"
DEFAULT_DISTANCE_THRESHOLD = 0.4  # Augmenté pour plus de tolérance (0.35 était trop strict)
# Note: Les casquettes, barbes longues et lunettes peuvent réduire la précision de détection
# Pour améliorer: utiliser plusieurs photos de la même personne avec/sans accessoires

def _pil_crop_and_save(src_path, bbox, dst_path):
    img = Image.open(src_path).convert("RGB")
    left, top, right, bottom = bbox
    crop = img.crop((left, top, right, bottom))
    crop.save(dst_path, format='JPEG', quality=90)
    return dst_path

def _try_deepface_represent(img_input: Any, model_name: str = MODEL_NAME, detector_backend: str = None) -> Optional[List[float]]:
    """
    Call DeepFace.represent with retries over provided backend if detector_backend is None.
    img_input may be a filesystem path or a numpy array (RGB).
    Returns embedding list or None.
    """
    if not DEEPFACE_AVAILABLE:
        logger.warning("DeepFace not available, cannot compute embeddings")
        return None
        
    backends = [detector_backend] if detector_backend else PREFERRED_BACKENDS
    last_exc = None
    
    for backend in backends:
        try:
            logger.debug(f"Trying DeepFace.represent with backend: {backend}")
            
            # S'assurer que l'image numpy est en uint8 (0-255) - DeepFace attend ce format
            if isinstance(img_input, np.ndarray):
                if img_input.dtype != np.uint8:
                    # Si l'image est en float (0-1), la reconvertir en uint8 (0-255)
                    if img_input.max() <= 1.0:
                        img_input = (img_input * 255).astype(np.uint8)
                    else:
                        img_input = img_input.astype(np.uint8)
                
            emb = DeepFace.represent(
                img_path=img_input,
                model_name=model_name,
                detector_backend=backend,
                enforce_detection=False,
                align=True  # Améliore la précision
            )
            
            logger.debug(f"DeepFace.represent succeeded with backend: {backend}")
            
            # Normalize outputs to flat list
            if emb is None:
                return None
                
            # Gérer les différents formats de retour de DeepFace
            result_embedding = None
            
            if isinstance(emb, list) and len(emb) > 0:
                if isinstance(emb[0], dict) and 'embedding' in emb[0]:
                    # Format DeepFace récent
                    embedding = emb[0]['embedding']
                    result_embedding = [float(x) for x in embedding]
                elif isinstance(emb[0], (list, tuple, np.ndarray)):
                    # Format ancien
                    result_embedding = [float(x) for x in emb[0]]
                elif isinstance(emb[0], (int, float)):
                    # Format simple
                    result_embedding = [float(x) for x in emb]
            elif isinstance(emb, (list, tuple, np.ndarray)):
                # Format direct
                result_embedding = [float(x) for x in emb]
            
            if result_embedding:
                # Vérifier que l'embedding est valide (non-nul)
                emb_array = np.array(result_embedding)
                emb_norm = np.linalg.norm(emb_array)
                emb_mean = np.mean(emb_array)
                logger.debug(f"Embedding computed: size={len(result_embedding)}, norm={emb_norm:.3f}, mean={emb_mean:.6f}")
                
                if emb_norm == 0:
                    logger.error("Embedding is all zeros! This is invalid.")
                    return None
                    
                return result_embedding
                
            logger.warning(f"Unexpected embedding format from DeepFace: {type(emb)}")
            return None
            
        except Exception as e:
            last_exc = e
            # Filtrer les erreurs connues
            msg = str(e)
            if "KerasTensor" in msg:
                logger.warning(f"KerasTensor error with backend={backend}, trying next")
            elif "MediaPipe" in msg and "not installed" in msg:
                logger.warning(f"MediaPipe not installed, skipping backend={backend}")
            elif "Incompatible shapes" in msg:
                logger.warning(f"Shape incompatibility with backend={backend}, trying next")
            else:
                logger.warning(f"DeepFace.represent failed with backend={backend}: {msg[:100]}")
            continue
            
    logger.error(f"All DeepFace.represent attempts failed; last error: {str(last_exc)[:200] if last_exc else 'Unknown'}")
    return None

def compute_embedding_for_image_path(image_input: Any):
    """
    Compute embedding for a given image path or numpy array using DeepFace.represent.
    Returns a list of floats or None.
    """
    # Si c'est un chemin de fichier, vérifier les caractères spéciaux
    if isinstance(image_input, str):
        try:
            # Essayer de lire l'image avec PIL pour contourner le problème des caractères spéciaux
            from PIL import Image
            img = Image.open(image_input).convert('RGB')
            img_array = np.array(img)
            return _try_deepface_represent(img_input=img_array)
        except Exception as e:
            logger.warning(f"Failed to load image with PIL, trying direct path: {e}")
            return _try_deepface_represent(img_input=image_input)
    else:
        return _try_deepface_represent(img_input=image_input)

def _try_deepface_extract_faces(img_path: str, detector_backend: Optional[str] = None) -> List[Dict]:
    """
    Call DeepFace.extract_faces with retries over backends.
    Returns DeepFace.extract_faces result ou liste vide.
    """
    if not DEEPFACE_AVAILABLE:
        logger.warning("DeepFace not available, using OpenCV fallback")
        return _opencv_extract_faces(img_path)
        
    backends = [detector_backend] if detector_backend else PREFERRED_BACKENDS
    last_exc = None
    
    for backend in backends:
        try:
            logger.debug(f"Trying DeepFace.extract_faces with backend: {backend}")
            
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=backend,
                enforce_detection=False,
                align=True,
                expand_percentage=15  # Augmenter la zone de détection pour capturer plus de contexte
            )
            
            if faces and len(faces) > 0:
                logger.debug(f"DeepFace.extract_faces found {len(faces)} faces with backend: {backend}")
                return faces
            else:
                logger.debug(f"No faces found with backend: {backend}")
                return []
                
        except Exception as e:
            last_exc = e
            msg = str(e)
            if "KerasTensor" in msg:
                logger.warning(f"KerasTensor error with backend={backend}, trying next")
            elif "MediaPipe" in msg and "not installed" in msg:
                logger.warning(f"MediaPipe not installed, skipping backend={backend}")
            elif "Incompatible shapes" in msg:
                logger.warning(f"Shape incompatibility with backend={backend}, trying next")
            else:
                logger.warning(f"DeepFace.extract_faces failed with backend={backend}: {msg[:100]}")
            continue
            
    logger.error(f"All DeepFace.extract_faces attempts failed; using OpenCV fallback. Last error: {str(last_exc)[:200] if last_exc else 'Unknown'}")
    return _opencv_extract_faces(img_path)

def _opencv_extract_faces(img_path: str) -> List[Dict]:
    """
    Fallback face detection using OpenCV Haar cascades.
    Returns list of face dicts compatible with DeepFace format.
    """
    try:
        # Charger le classificateur Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Could not read image: {img_path}")
            return []
            
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = []
        for (x, y, w, h) in faces:
            # Extraire la région du visage
            face_img = img[y:y+h, x:x+w]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            result.append({
                'face': face_img_rgb,
                'facial_area': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'confidence': 1.0  # OpenCV ne fournit pas de confiance
            })
            
        logger.info(f"OpenCV detected {len(result)} faces in {img_path}")
        return result
        
    except Exception as e:
        logger.exception(f"OpenCV face detection failed for {img_path}: {e}")
        return []

def build_gallery_embeddings(force=False):
    """
    Compute and store embeddings for all Photo objects and Profile.photo_user if missing or force=True.
    """
    photos = Photo.objects.all()
    for p in photos:
        if not p.embedding or force:
            if p.image:
                try:
                    emb = compute_embedding_for_image_path(p.image.path)
                    if emb is not None:
                        p.embedding = emb
                        p.save(update_fields=['embedding'])
                        logger.info("Indexed embedding for Photo id=%s person=%s", p.pk, p.person_name)
                except Exception as e:
                    logger.exception("Failed to index Photo %s: %s", p.pk, e)

    # Profiles photos
    profiles = Profile.objects.exclude(photo_user=None).exclude(photo_user='')
    for prof in profiles:
        try:
            if prof.photo_user:
                emb = compute_embedding_for_image_path(prof.photo_user.path)
                if emb:
                    album, _ = PhotoAlbum.objects.get_or_create(
                        user=prof.user,
                        name='__profiles__',
                        defaults={'description': 'Auto-generated profiles album'}
                    )
                    photo_qs = Photo.objects.filter(album=album, person_name=prof.user.username)
                    if photo_qs.exists():
                        photo = photo_qs.first()
                        photo.embedding = emb
                        photo.save(update_fields=['embedding'])
                    else:
                        Photo.objects.create(
                            album=album,
                            image=prof.photo_user,
                            person_name=prof.user.username,
                            is_user=True,
                            embedding=emb
                        )
                        logger.info("Created profile Photo for user %s", prof.user.username)
        except Exception as e:
            logger.exception("Error indexing profile %s: %s", getattr(prof, 'pk', 'unknown'), e)

def _gather_indexed_embeddings():
    """
    Returns list of dicts: {'person_name':..., 'embedding': np.array, 'photo_id':..., 'is_user': bool, 'priority': int}
    Only includes Photo objects with embedding present.
    Photos de profil utilisateur ont la priorité la plus élevée.
    """
    items = []
    for p in Photo.objects.exclude(embedding__isnull=True).exclude(embedding__exact=''):
        try:
            emb = np.array(p.embedding, dtype=np.float32)
            
            # Déterminer la priorité : 1 = profil utilisateur (le plus important), 2 = photo d'album
            priority = 1 if p.is_user and p.album.name == '__profiles__' else 2
            
            items.append({
                'person_name': p.person_name, 
                'embedding': emb, 
                'photo_id': p.pk,
                'is_user': p.is_user,
                'priority': priority
            })
        except Exception:
            logger.exception("Bad embedding for Photo %s", p.pk)
    
    # Trier par priorité (profils utilisateur en premier)
    items.sort(key=lambda x: x['priority'])
    
    logger.info(f"Gallery loaded: {len(items)} embeddings (priority sorted)")
    return items

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance between two vectors (0 = same, 1 = opposite)."""
    if a is None or b is None:
        return 1.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    cos_sim = np.dot(a, b) / (a_norm * b_norm)
    cos_sim = max(min(cos_sim, 1.0), -1.0)
    return 1.0 - cos_sim

def _distance_to_confidence(dist):
    dist = max(0.0, min(1.0, float(dist)))
    conf = max(0.0, (1.0 - dist)) * 100.0
    return round(conf, 1)

def detect_faces(image_path: str) -> List[Dict]:
    """
    Detect faces using DeepFace.extract_faces with backend fallbacks.
    Returns list of dicts: {'face': np.ndarray_or_path, 'location': {'left','top','right','bottom'}}
    """
    faces = _try_deepface_extract_faces(image_path)
    results = []
    if not faces:
        logger.info(f"No faces detected in {image_path}")
        return results
        
    logger.info(f"Processing {len(faces)} detected faces from {image_path}")
    
    for i, f in enumerate(faces):
        try:
            facial_area = f.get('facial_area') or {}
            x = int(facial_area.get('x', 0))
            y = int(facial_area.get('y', 0))
            w = int(facial_area.get('w', 0))
            h = int(facial_area.get('h', 0))
            left = x
            top = y
            right = x + w
            bottom = y + h
            
            # Validation des coordonnées
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid face dimensions: w={w}, h={h} for face {i}")
                continue
                
            results.append({
                'face': f.get('face'),
                'location': {'left': left, 'top': top, 'right': right, 'bottom': bottom},
                'confidence': f.get('confidence', 0.0)
            })
            
        except Exception as e:
            logger.exception(f"Error processing face {i}: {e}")
            continue
            
    logger.info(f"Successfully processed {len(results)} faces from {image_path}")
    return results

def compare_with_users(image_path: str, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD) -> List[Dict]:
    """
    Detect faces with DeepFace in image_path, compute embeddings and compare against indexed gallery.
    Returns list of faces with location + match info.
    """
    logger.info(f"Starting face comparison for {image_path}")
    
    try:
        # Étape 1: Détecter les visages
        detected = detect_faces(image_path)
        if not detected:
            logger.info(f"No faces detected in {image_path}")
            return []

        # Étape 2: Construire la galerie de visages connus
        gallery = _gather_indexed_embeddings()
        if not gallery:
            logger.warning("No indexed faces found in gallery for comparison")
            # Retourner les visages détectés sans identification
            return [{
                'location': item['location'],
                'username': None,
                'person_name': None,
                'confidence': 0.0,
                'photo_id': None,
            } for item in detected]

        gallery_embeddings = np.array([g['embedding'] for g in gallery], dtype=np.float32)
        logger.info(f"Gallery contains {len(gallery)} indexed faces")

        # Étape 3: Comparer chaque visage détecté
        results = []
        for i, item in enumerate(detected):
            loc = item['location']
            face_img = item['face']  # numpy array (RGB) ou PIL image
            
            logger.debug(f"Processing face {i+1}/{len(detected)} at location {loc}")
            
            # Calculer l'embedding pour le visage détecté
            enc = _try_deepface_represent(img_input=face_img)
            
            if enc is None:
                logger.warning(f"Could not compute embedding for face {i+1}")
                results.append({
                    'location': loc,
                    'username': None,
                    'person_name': None,
                    'confidence': 0.0,
                    'photo_id': None,
                })
                continue
                
            # Comparer avec la galerie - algorithme amélioré avec pondération
            enc_arr = np.array(enc, dtype=np.float32)
            
            # Vérifier que l'embedding du visage détecté est valide
            enc_norm = np.linalg.norm(enc_arr)
            logger.debug(f"Face {i+1} embedding: norm={enc_norm:.3f}, mean={np.mean(enc_arr):.6f}")
            
            if enc_norm == 0:
                logger.error(f"Face {i+1} has zero embedding! Skipping comparison.")
                results.append({
                    'location': loc,
                    'username': None,
                    'person_name': None,
                    'confidence': 0.0,
                    'photo_id': None,
                })
                continue
            
            # Calculer les distances pour tous les candidats
            candidates = []
            for g in gallery:
                try:
                    dist = _cosine_distance(enc_arr, g['embedding'])
                    # Appliquer un bonus aux profils utilisateur (réduire légèrement la distance)
                    adjusted_dist = dist * 0.95 if g['priority'] == 1 else dist
                    candidates.append({
                        'match': g,
                        'distance': dist,
                        'adjusted_distance': adjusted_dist,
                        'priority': g['priority']
                    })
                except Exception as e:
                    logger.warning(f"Error computing distance for face {i+1}: {e}")
                    continue
            
            if not candidates:
                logger.warning(f"Face {i+1}: no valid candidates")
                results.append({
                    'location': loc,
                    'username': None,
                    'person_name': None,
                    'confidence': 0.0,
                    'photo_id': None,
                })
                continue
            
            # Trier par distance ajustée
            candidates.sort(key=lambda x: x['adjusted_distance'])
            
            # Prendre le meilleur candidat
            best_candidate = candidates[0]
            best_distance = best_candidate['distance']
            best_match = best_candidate['match']
            
            # Log des 3 meilleurs candidats pour debug
            logger.info(f"Face {i+1} - Top 3 candidates:")
            for idx, cand in enumerate(candidates[:3]):
                logger.info(f"  {idx+1}. {cand['match']['person_name']}: distance={cand['distance']:.3f}, priority={cand['priority']}, is_user={cand['match']['is_user']}")
            
            # Vérifier si la correspondance est suffisamment bonne
            # Seuil plus strict pour les profils non-utilisateur
            effective_threshold = distance_threshold if best_match['priority'] == 1 else distance_threshold * 0.85
            
            if best_distance <= effective_threshold:
                confidence = _distance_to_confidence(best_distance)
                logger.info(f"Face {i+1} MATCHED with {best_match['person_name']} (distance: {best_distance:.3f}, confidence: {confidence}%, priority: {best_match['priority']})")
                
                results.append({
                    'location': loc,
                    'username': best_match['person_name'],
                    'person_name': best_match['person_name'],
                    'confidence': confidence,
                    'photo_id': best_match['photo_id'],
                })
            else:
                logger.info(f"Face {i+1} NO MATCH (best: {best_match['person_name']} with distance {best_distance:.3f} > threshold {effective_threshold:.3f})")
                results.append({
                    'location': loc,
                    'username': None,
                    'person_name': None,
                    'confidence': 0.0,
                    'photo_id': None,
                })

        logger.info(f"Face comparison completed for {image_path}: {len(results)} faces processed")
        return results
        
    except Exception as e:
        logger.exception(f"compare_with_users error for {image_path}: {e}")
        return []