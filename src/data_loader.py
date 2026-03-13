import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class OilDataLoader:
    """
    Module pour charger les données historiques du pétrole.
    """
    def __init__(self, symbol: str = "CL=F"):
        self.symbol = symbol

    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Récupère les données historiques via Yahoo Finance.
        """
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"Aucune donnée trouvée pour le symbole {self.symbol}")
            return df
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des indicateurs techniques simples (Moyennes mobiles, RSI).
        """
        df = df.copy()
        # Moyennes Mobiles
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
