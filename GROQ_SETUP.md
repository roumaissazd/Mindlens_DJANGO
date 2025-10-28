# Configuration Groq AI (Gratuit et Rapide)

## 🎯 Pourquoi Groq ?
- **100% GRATUIT** - Aucune carte de crédit nécessaire
- **Très RAPIDE** - Réponses en < 1 seconde
- **Sans limite** - Utilisez autant que vous voulez
- **Pas de serveur local** - Tout dans le cloud

## 📋 Étapes d'installation

### 1. Installer Groq
```bash
pip install groq
```

### 2. Obtenir votre clé API (GRATUIT)

1. Allez sur https://console.groq.com/
2. Créez un compte (c'est gratuit, pas de carte)
3. Dans "API Keys", créez une nouvelle clé
4. Copiez la clé

### 3. Configurer la clé

#### Option A : Variable d'environnement (Recommandé)
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="votre-clé-ici"

# Windows (CMD)
set GROQ_API_KEY=votre-clé-ici

# Linux/Mac
export GROQ_API_KEY="votre-clé-ici"
```

#### Option B : Fichier .env
Créez un fichier `.env` à la racine du projet :
```env
GROQ_API_KEY=votre-clé-ici
```

### 4. Démarrer MindLense
```bash
python manage.py runserver
```

## ✨ C'est tout !
Le système détectera automatiquement si Groq est disponible et utilisera l'IA pour répondre intelligemment.

## 🆘 Si Groq n'est pas disponible
Le système revient automatiquement au mode règles (comme avant) - tout fonctionne quand même !

## 💡 Alternative : Ollama (Local)
Si vous préférez une solution 100% locale :
```bash
# Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Télécharger un modèle
ollama pull llama3.1:8b
```

Puis modifiez le code pour utiliser Ollama au lieu de Groq.

## 🚀 Résultat
Maintenant votre agent vocal :
- ✅ Comprend n'importe quelle demande de l'utilisateur
- ✅ Répond intelligemment sans règles manuelles
- ✅ S'adapte naturellement à la conversation
- ✅ Garde le contexte de la conversation

**Exemple :**
- Utilisateur : "Je suis stressé, je veux écouter du jazz"
- Agent : "Je comprends. Voici un extrait musical apaisant."
- *[Musique démarre]*

Plus besoin de coder des règles pour chaque cas !







