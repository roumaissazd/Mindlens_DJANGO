MindLense
============

Plateforme de journaling personnel multimodal avec IA (Django).

Demarrage
---------

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install django
python manage.py migrate
python manage.py runserver 127.0.0.1:8010
```

Auth
----
- /signup
- /login
- /logout

Developpement
-------------
- Templates: templates/
- Static: static/
- App: core/


Voice Journal (local ASR/Sentiment/Emotion)
-------------------------------------------

Prérequis Windows (FFmpeg):
- Téléchargez l'archive ZIP de FFmpeg (build statique) et extrayez-la dans `C:\ffmpeg`.
- Ajoutez `C:\ffmpeg\bin` à votre PATH système, ouvrez un nouveau PowerShell, et vérifiez:

```bash
ffmpeg -version
```

Installation (dans le dossier du projet):

```powershell
# Optionnel: utilisez un venv dédié
python -m venv venv
.\venv\Scripts\Activate.ps1

# Dépendances Python
pip install -r requirements.txt

# Torch CPU (Windows):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Lancement du serveur:

```powershell
python manage.py migrate
python manage.py runserver 8000
```

Accès:
- Ouvrez `http://127.0.0.1:8000/voice-journal/`

Notes:
- Premier lancement: le modèle Whisper tiny et le modèle de sentiment seront téléchargés (10–60s). Les appels suivants sont plus rapides.
- Gardez les premiers enregistrements courts (5–8s) pour tester.
- Aucun compte requis; démo stateless, fichiers `.wav` sauvegardés sous `media/audio/`.
- En cas d'erreur d'import ML/audio, la réponse JSON contient `missing: [ ... ]`.
- Sous Windows, assurez-vous que FFmpeg est bien sur le PATH pour `pydub`.


