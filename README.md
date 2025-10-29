MindLense
============

Plateforme de journaling personnel multimodal avec IA (Django).

Demarrage
---------

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install django
pip install transformers
pip install torch librosa pydub

pip install Whoosh
pip install pillow

pip install deepface
pip install tensorflow
pip install opencv-python
pip install retina-face
pip install tf-keras



pip install gTTS
pip install langdetect




python manage.py createsuperuser

python manage.py verify_images

python manage.py makemigrations 
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


