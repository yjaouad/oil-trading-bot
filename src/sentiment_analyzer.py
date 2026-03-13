import feedparser
from transformers import pipeline
import pandas as pd
from typing import List, Dict

class SentimentAnalyzer:
    """
    Module pour l'analyse de sentiment et la détection des tensions géopolitiques.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # Initialisation du pipeline de sentiment IA
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            print(f"Erreur d'initialisation du modèle IA: {e}")
            self.sentiment_pipeline = None

        # Mots-clés pour les tensions géopolitiques
        self.geopolitical_keywords = [
            "war", "conflict", "sanction", "OPEC", "embargo", 
            "Middle East", "Russia", "Ukraine", "Iran", "tensions", 
            "supply", "disruption", "military", "crisis"
        ]

    def fetch_oil_news(self) -> List[Dict]:
        """
        Récupère les actualités du pétrole via plusieurs flux RSS.
        """
        rss_urls = [
            "https://www.cnbc.com/id/19835149/device/rss/rss.html", # CNBC Oil
            "https://finance.yahoo.com/rss/headline?s=CL=F",       # Yahoo Oil
            "https://www.reutersagency.com/feed/?best-topics=energy&format=xml" # Reuters Energy
        ]
        
        news_items = []
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]: # On prend les 5 premières de chaque
                    news_items.append({
                        "title": entry.title,
                        "summary": entry.summary if 'summary' in entry else entry.title,
                        "link": entry.link,
                        "published": entry.published if 'published' in entry else "N/A"
                    })
            except Exception as e:
                print(f"Erreur flux RSS {url}: {e}")
                
        return news_items

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyse le sentiment d'un texte via le modèle IA.
        """
        if not self.sentiment_pipeline:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            # On limite la longueur pour éviter les erreurs de tokenisation
            result = self.sentiment_pipeline(text[:512])[0]
            return result
        except Exception as e:
            print(f"Erreur d'analyse sentiment: {e}")
            return {"label": "NEUTRAL", "score": 0.5}

    def detect_geopolitical_risk(self, text: str) -> bool:
        """
        Détecte si un texte contient des mots-clés liés aux tensions géopolitiques.
        """
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.geopolitical_keywords)

    def process_news(self) -> pd.DataFrame:
        """
        Récupère et analyse un ensemble d'actualités.
        """
        news = self.fetch_oil_news()
        results = []
        seen_titles = set()
        columns = ["Date", "Headline", "Sentiment", "Confidence", "Geopolitical Risk", "Link"]
        
        for item in news:
            if item['title'] in seen_titles:
                continue
            seen_titles.add(item['title'])
            
            sentiment = self.analyze_sentiment(item['title'] + " " + item['summary'])
            is_geopolitical = self.detect_geopolitical_risk(item['title'] + " " + item['summary'])
            
            results.append({
                "Date": item['published'],
                "Headline": item['title'],
                "Sentiment": sentiment['label'],
                "Confidence": sentiment['score'],
                "Geopolitical Risk": is_geopolitical,
                "Link": item['link']
            })
            
        if not results:
            return pd.DataFrame(columns=columns)
            
        return pd.DataFrame(results)
