import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict

class TrendPredictor:
    """
    Module pour la prédiction des tendances de prix via Machine Learning sur plusieurs horizons.
    """
    def __init__(self, model_params: Dict = None):
        if model_params is None:
            model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        self.model_1d = RandomForestRegressor(**model_params)
        self.model_7d = RandomForestRegressor(**model_params)
        self.model_30d = RandomForestRegressor(**model_params)
        self.features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'Volume']

    def prepare_data(self, df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les données pour un horizon spécifique (en jours).
        """
        X = df[self.features].copy()
        y = df['Close'].shift(-horizon)
        
        # Enlever les lignes avec des labels NaN (fin de série)
        X = X[:-horizon]
        y = y[:-horizon]
        
        return X, y

    def train_all_models(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entraîne les modèles pour 1j, 7j et 30j.
        """
        metrics = {}
        horizons = {"1d": 1, "7d": 7, "30d": 30}
        models = {"1d": self.model_1d, "7d": self.model_7d, "30d": self.model_30d}

        for name, h in horizons.items():
            X, y = self.prepare_data(df, h)
            if len(X) < 50:
                continue
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models[name].fit(X_train, y_train)
            
            y_pred = models[name].predict(X_test)
            metrics[name] = np.sqrt(mean_squared_error(y_test, y_pred))
            
        return metrics

    def predict_horizons(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Prédit les prix pour les 3 horizons si les modèles sont entraînés.
        """
        last_day_data = df[self.features].tail(1)
        predictions = {}
        
        horizons = {"1d": self.model_1d, "7d": self.model_7d, "30d": self.model_30d}
        
        for name, model in horizons.items():
            try:
                # Vérifie si le modèle est entraîné (RandomForest a l'attribut estimators_ après fit)
                if hasattr(model, 'estimators_'):
                    predictions[name] = model.predict(last_day_data)[0]
                else:
                    predictions[name] = 0.0
            except:
                predictions[name] = 0.0
                
        return predictions

    def get_trend_signal(self, current_price: float, predicted_price: float) -> str:
        """
        Génère un signal (ACHAT, VENTE, NEUTRE) selon la prédiction.
        """
        change_pct = (predicted_price - current_price) / current_price * 100
        if change_pct > 1.5:
            return "HAUSSIER (ACHAT FORT)"
        elif change_pct > 0.5:
            return "LÉGÈRE HAUSSE (ACHAT)"
        elif change_pct < -1.5:
            return "BAISSIER (VENTE FORTE)"
        elif change_pct < -0.5:
            return "LÉGÈRE BAISSE (VENTE)"
        else:
            return "NEUTRE"
