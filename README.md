# 🛢️ PetroTrend IA - Bot de Trading Pétrole

PetroTrend IA est un bot de trading professionnel capable de prévoir les tendances du pétrole, d'analyser le sentiment du marché et de détecter les risques géopolitiques en utilisant l'IA et le Machine Learning.

## 🚀 Comment déployer comme un site web (Gratuit)

La méthode la plus simple et la plus rapide pour déployer cette application est d'utiliser **Streamlit Community Cloud**.

### 1. Préparer votre compte GitHub
- Créez un nouveau dépôt (repository) sur [GitHub](https://github.com/).
- Envoyez tous les fichiers du projet (sauf le dossier `__pycache__` et le fichier `.env`) vers ce dépôt.

### 2. Déployer sur Streamlit Cloud
1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io/).
2. Connectez-vous avec votre compte GitHub.
3. Cliquez sur **"New app"**.
4. Sélectionnez votre dépôt, la branche (généralement `main`) et le fichier principal : `app.py`.
5. Cliquez sur **"Deploy!"**.

### 3. Gérer vos Clés API (Secrets)
Si vous utilisez des clés API (comme `NEWS_API_KEY` dans votre `.env`), ne les envoyez pas sur GitHub !
- Sur votre tableau de bord Streamlit Cloud, allez dans **Settings > Secrets**.
- Copiez le contenu de votre fichier `.env` dans la zone de texte sous ce format :
  ```toml
  NEWS_API_KEY = "votre_cle_ici"
  ```

## 🛠️ Installation locale
```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## 📊 Fonctionnalités
- **Analyse de Sentiment IA** : Modèles Transformers pour l'analyse des news.
- **Machine Learning Multi-Horizons** : Prévisions à 24h, 7j et 30j via Random Forest.
- **Dashboard Interactif** : Graphiques de prix, volumes et tendances prédictives.
- **Risques Géopolitiques** : Détection automatique des tensions mondiales.

---
*Avertissement : Cet outil est à but éducatif. Le trading comporte des risques financiers réels.*
