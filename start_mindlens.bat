@echo off
echo ========================================
echo    DEMARRAGE AUTOMATIQUE MINDLENSE
echo ========================================
echo.

REM Aller dans le dossier du projet
cd /d "D:\mindlens"

REM Activer l'environnement virtuel
echo [1/3] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

REM Vérifier que whoosh est installé
echo [2/3] Verification des dependances...
python -c "import whoosh; print('Whoosh OK')" 2>nul || (
    echo Installation de whoosh...
    pip install whoosh
)

REM Démarrer le serveur
echo [3/3] Demarrage du serveur Django...
echo.
echo ✅ Serveur accessible sur : http://127.0.0.1:8010/
echo ✅ Appuyez sur Ctrl+C pour arreter le serveur
echo.
python manage.py runserver 127.0.0.1:8010

pause
