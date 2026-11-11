"""Tableau de bord Streamlit : historique, prédictions, alertes et score de drift.

Version initiale : les valeurs viennent de données de démonstration. Elles seront
remplacées par les réponses de l'API d'inférence (/predict, /drift-score).
Lancement : streamlit run src/dashboard.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.alerts import SATURATION_THRESHOLD, count_alerts, saturation_alert


def demo_series(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Série de démonstration : affluence observée et prédite."""
    rng = np.random.default_rng(seed)
    observed = np.cumsum(rng.normal(0.0, 0.3, n)) + 2.0
    predicted = observed + rng.normal(0.0, 0.2, n)
    return pd.DataFrame({"observée": observed, "prédite": predicted})


st.set_page_config(page_title="Adaptive Crowd Monitoring", layout="wide")
st.title("Adaptive Crowd Monitoring")
st.caption("Prédiction de l'affluence et détection de drift (données de démonstration)")

data = demo_series()
last = float(data["prédite"].iloc[-1])
level = saturation_alert(last)

col1, col2, col3 = st.columns(3)
col1.metric("Affluence prédite", f"{last:.2f}")
col2.metric("Score de drift", "--")          # sera fourni par /drift-score
col3.metric("Niveau d'alerte", level)

if level == "saturation":
    st.error("Alerte : saturation prévue.")
elif level == "attention":
    st.warning("Attention : densité proche du seuil.")
else:
    st.success("Situation normale.")

st.subheader("Historique et prédictions")
st.line_chart(data)
st.write(f"Pas de temps au-dessus du seuil ({SATURATION_THRESHOLD}) : "
         f"{count_alerts(data['prédite'])}")