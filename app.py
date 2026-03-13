import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data_loader import OilDataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.predictor import TrendPredictor

# Configuration de la page
st.set_page_config(
    page_title="PetroTrend IA - Bot de Trading Pétrole Professionnel",
    layout="wide",
    page_icon="🛢️"
)

# Style personnalisé
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1.1rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    /* Amélioration de la visibilité des onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des modules (Cachés pour performance)
@st.cache_resource
def init_trading_bot():
    loader = OilDataLoader()
    analyzer = SentimentAnalyzer()
    predictor = TrendPredictor()
    return loader, analyzer, predictor

loader, analyzer, predictor = init_trading_bot()

# --- SIDEBAR ---
st.sidebar.title("Paramètres 🛢️")
oil_type = st.sidebar.selectbox("Type de Pétrole", ["WTI (CL=F)", "Brent (BZ=F)"])
period = st.sidebar.selectbox("Période historique", ["6mo", "1y", "2y", "5y"], index=1)
update_btn = st.sidebar.button("Actualiser les données")

# --- HEADER ---
st.title("🛢️ PetroTrend IA - Analyseur de Marché Pétrolier")
st.subheader("Analyse prédictive des tendances, du sentiment et des risques géopolitiques.")

# --- CHARGEMENT DES DONNÉES ---
with st.spinner("Chargement des données du marché..."):
    loader.symbol = oil_type.split("(")[1].split(")")[0]
    raw_df = loader.get_historical_data(period=period)
    df = loader.add_technical_indicators(raw_df)

if df.empty:
    st.error("Erreur: Impossible de charger les données boursières.")
    st.stop()

# --- ANALYSE IA (Sentiment & Tensions) ---
with st.spinner("Analyse du sentiment IA et détection des risques..."):
    news_df = analyzer.process_news()
    if not news_df.empty:
        geopolitical_risk_detected = news_df['Geopolitical Risk'].any()
        avg_sentiment_score = news_df['Confidence'].mean()
        sentiment_label = "POSITIF" if avg_sentiment_score > 0.6 else "NÉGATIF" if avg_sentiment_score < 0.4 else "NEUTRE"
    else:
        geopolitical_risk_detected = False
        avg_sentiment_score = 0.5
        sentiment_label = "INDISPONIBLE"

# --- PRÉDICTION IA ---
current_price = df['Close'].iloc[-1]
with st.spinner("Génération de la prédiction des tendances..."):
    try:
        metrics = predictor.train_all_models(df)
        predictions = predictor.predict_horizons(df)
        # S'assurer que toutes les clés existent
        for h in ["1d", "7d", "30d"]:
            if h not in predictions:
                predictions[h] = 0.0
        
        signal = predictor.get_trend_signal(current_price, predictions['1d']) if predictions['1d'] > 0 else "INDÉTERMINÉ"
    except Exception as e:
        st.warning(f"Prédiction ML indisponible: {e}")
        predictions = {"1d": 0.0, "7d": 0.0, "30d": 0.0}
        signal = "INDÉTERMINÉ"

# --- DASHBOARD METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Prix Actuel", f"{current_price:.2f}$", f"{df['Close'].diff().iloc[-1]:.2f}$")
with col2:
    st.metric("Sentiment IA", sentiment_label, f"{avg_sentiment_score*100:.1f}% Confiance")
with col3:
    risk_color = "🔴" if geopolitical_risk_detected else "🟢"
    st.metric("Risque Géopolitique", "ÉLEVÉ" if geopolitical_risk_detected else "FAIBLE", risk_color)
with col4:
    st.metric("Signal IA", signal)

# --- CHARTS ---
st.markdown("---")
tab_charts, tab_volume, tab_prediction = st.tabs(["📈 Analyse de Prix", "📊 Volume des Échanges", "🔮 Prévisions IA"])

with tab_charts:
    st.subheader("Analyse Graphique et Technique")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Prix du Pétrole"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='blue', width=1), name="SMA 50"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab_volume:
    st.subheader("Volume des Transactions (Estimation Achat/Vente)")
    # Calcul simple pour estimer achat vs vente basé sur la direction du prix
    df['Vol_Color'] = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(df['Open'], df['Close'])]
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=df.index, 
        y=df['Volume'],
        marker_color=df['Vol_Color'],
        name="Volume"
    ))
    fig_vol.update_layout(
        height=400, 
        template="plotly_dark",
        yaxis_title="Volume",
        showlegend=False
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.caption("🟢 Volume acheteur (Prix ↑) | 🔴 Volume vendeur (Prix ↓)")

with tab_prediction:
    st.subheader("Graphique Prévisionnel IA")
    # Préparation des données de prédiction
    last_date = df.index[-1]
    pred_dates = [last_date + timedelta(days=1), last_date + timedelta(days=7), last_date + timedelta(days=30)]
    pred_values = [predictions['1d'], predictions['7d'], predictions['30d']]
    
    # On ne garde que les prédictions valides (> 0)
    valid_indices = [i for i, v in enumerate(pred_values) if v > 0]
    plot_dates = [pred_dates[i] for i in valid_indices]
    plot_values = [pred_values[i] for i in valid_indices]
    
    fig_pred = go.Figure()
    # Historique récent (30 derniers jours)
    hist_subset = df.tail(30)
    fig_pred.add_trace(go.Scatter(x=hist_subset.index, y=hist_subset['Close'], name="Historique", line=dict(color='white')))
    
    if plot_values:
        # Ligne de tendance prédictive
        fig_pred.add_trace(go.Scatter(
            x=[df.index[-1]] + plot_dates, 
            y=[df['Close'].iloc[-1]] + plot_values,
            name="Prédiction IA",
            line=dict(color='#007bff', dash='dash', width=3),
            mode='lines+markers'
        ))
    
    fig_pred.update_layout(height=500, template="plotly_dark", yaxis_title="Prix ($)")
    st.plotly_chart(fig_pred, use_container_width=True)

# --- NEWS & SENTIMENT SECTION ---
st.markdown("---")
col_news, col_pred = st.columns([2, 1])

with col_news:
    st.subheader("Dernières Actualités & Sentiment IA")
    if news_df.empty:
        st.info("Aucune actualité récente trouvée pour le moment.")
    else:
        for _, row in news_df.iterrows():
            sentiment_icon = "🟢" if row['Sentiment'] == 'POSITIVE' else "🔴"
            geo_icon = "⚠️ Tensions détectées" if row['Geopolitical Risk'] else ""
            st.markdown(f"""
            **{sentiment_icon} {row['Headline']}**  
            *Publié le: {row['Date']}* | Sentiment: {row['Sentiment']} ({row['Confidence']:.1f}%) {geo_icon}  
            [Lire la suite]({row['Link']})
            """)
            st.markdown("---")

with col_pred:
    st.subheader("📊 Rapport de Prédiction IA")
    st.write(f"**Prix Actuel:** `{current_price:.2f}$`")
    
    # Onglets pour les différents horizons
    tab1, tab2, tab3 = st.tabs(["24 Heures", "1 Semaine", "1 Mois"])
    
    with tab1:
        pred_1d = predictions['1d']
        if pred_1d > 0:
            change_1d = ((pred_1d - current_price) / current_price) * 100
            st.metric("Prix Prédit (24h)", f"{pred_1d:.2f}$", f"{change_1d:+.2f}%")
            st.info(f"Signal: **{predictor.get_trend_signal(current_price, pred_1d)}**")
        else:
            st.write("Données insuffisantes")
            
    with tab2:
        pred_7d = predictions.get('7d', 0)
        if pred_7d > 0:
            change_7d = ((pred_7d - current_price) / current_price) * 100
            st.metric("Prix Prédit (7j)", f"{pred_7d:.2f}$", f"{change_7d:+.2f}%")
            st.info(f"Signal: **{predictor.get_trend_signal(current_price, pred_7d)}**")
        else:
            st.warning("Historique insuffisant pour prédire à 1 semaine. Sélectionnez une période plus longue (ex: 2y).")
            
    with tab3:
        pred_30d = predictions.get('30d', 0)
        if pred_30d > 0:
            change_30d = ((pred_30d - current_price) / current_price) * 100
            st.metric("Prix Prédit (30j)", f"{pred_30d:.2f}$", f"{change_30d:+.2f}%")
            st.info(f"Signal: **{predictor.get_trend_signal(current_price, pred_30d)}**")
        else:
            st.warning("Historique insuffisant pour prédire à 1 mois. Sélectionnez une période plus longue (ex: 5y).")

    st.markdown("---")
    st.write("💡 *Le modèle utilise une forêt aléatoire (Random Forest) entraînée sur les indicateurs techniques récents.*")
    
    if geopolitical_risk_detected:
        st.warning("⚠️ L'analyse IA a détecté des tensions géopolitiques actives. Prudence recommandée: ces facteurs peuvent entraîner une volatilité extrême non capturée par les modèles historiques.")

# Footer
st.markdown("---")
st.caption("Avertissement: Ce bot est un outil d'aide à la décision utilisant l'IA. Le trading comporte des risques. Les prédictions ne sont pas garanties.")
