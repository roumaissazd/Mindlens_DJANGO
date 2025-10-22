# Script de démarrage automatique MindLense
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    DEMARRAGE AUTOMATIQUE MINDLENSE" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Aller dans le dossier du projet
Set-Location "D:\mindlens"

# Activer l'environnement virtuel
Write-Host "[1/3] Activation de l'environnement virtuel..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Vérifier que whoosh est installé
Write-Host "[2/3] Vérification des dépendances..." -ForegroundColor Yellow
try {
    python -c "import whoosh; print('Whoosh OK')" 2>$null
    Write-Host "✅ Whoosh est installé" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Installation de whoosh..." -ForegroundColor Yellow
    pip install whoosh
}

# Démarrer le serveur
Write-Host "[3/3] Démarrage du serveur Django..." -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ Serveur accessible sur : http://127.0.0.1:8010/" -ForegroundColor Green
Write-Host "✅ Appuyez sur Ctrl+C pour arrêter le serveur" -ForegroundColor Green
Write-Host ""
python manage.py runserver 127.0.0.1:8010
