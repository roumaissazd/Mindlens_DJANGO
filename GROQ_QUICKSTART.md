# 🚀 Quick Start - Agent Vocal Intelligent

## ⚡ Installation en 2 minutes

### 1. Installer Groq
```bash
pip install groq
```

### 2. Obtenir votre API Key GRATUITE
1. Aller sur https://console.groq.com
2. Créer un compte (GRATUIT, pas de carte)
3. Copier la clé API

### 3. Configuration
```bash
# Windows PowerShell
$env:GROQ_API_KEY="gsk_votre_cle_ici"

# Démarrer
python manage.py runserver
```

### 4. Test
- Ouvrir https://localhost:8000/voice-journal
- Faire un enregistrement
- Cliquer sur "Commencer le soutien"
- Parler librement - l'agent comprend tout !

## 🎯 Comment ça marche maintenant

### Avant (Règles manuelles)
```python
# Il fallait coder pour chaque cas
if "musique" in texte:
    return "Voici de la musique"
elif "respiration" in texte:
    return "Faisons une respiration"
# ... 1000 lignes de règles
```

### Maintenant (IA intelligente)
```python
# L'IA comprend tout naturellement
reply = generate_llm_reply(user_text, mood, history)
# L'agent répond intelligemment selon le contexte
```

## 📝 Exemples d'interactions

### Exemple 1 : Demande directe
**User** : "Je suis anxieux, donne-moi de la musique"
**Agent** : "Extrait musical."
→ *Musique démarre automatiquement*

### Exemple 2 : Réponse simple
**User** : "Oui"
**Agent** : *Comprend le contexte et continue la conversation*

### Exemple 3 : Changement de sujet
**User** : "En fait j'aimerais plutôt faire une respiration"
**Agent** : "D'accord. On commence une respiration simple."
→ *Exercice de respiration démarre*

## 💡 Avantages

✅ **100% Gratuit** - Groq est gratuit sans limite
✅ **Très Rapide** - Réponses en <1 seconde
✅ **Intelligent** - Comprend le contexte
✅ **Naturel** - Pas besoin de phrases précises
✅ **Contexte** - Se souvient de la conversation

## 🆘 Dépannage

### "Module not found: groq"
```bash
pip install groq
```

### "No API key found"
```bash
# Vérifier que la variable d'environnement est définie
echo $env:GROQ_API_KEY  # PowerShell
echo %GROQ_API_KEY%     # CMD
```

### Fallback automatique
Si Groq n'est pas configuré, le système utilise les règles classiques - tout fonctionne !

## 🔧 Personnalisation

Vous pouvez changer le modèle dans `views.py` :
```python
model="llama-3.1-8b-instant"  # Rapide et gratuit
# Ou
model="mixtral-8x7b-32768"    # Plus puissant
```

## 🎉 C'est prêt !

Votre agent vocal est maintenant **intelligent** et **gratuit** !








