"""Airbnb Market Intelligence, Riviera du Levant (Guadeloupe)
Démonstration de soutenance.  Lancement :  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

ICI = Path(__file__).parent
ART = ICI / "artefacts"
# Charte MJ Metrix : corail, sarcelle, bleu, gris
BLEU, GRIS, ORANGE = "#D4532F", "#A0A2A8", "#2E8C80"
CORAIL, SARCELLE, AZUR, ENCRE = "#F37150", "#4EC0B0", "#20A2DC", "#373A3C"

st.set_page_config(page_title="Airbnb Market Intelligence", layout="wide")
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Alegreya+Sans:wght@700;800&family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, p, label, div[data-testid="stMetricValue"] {{ font-family: 'Poppins', Arial, sans-serif; }}
h1, h2, h3 {{ font-family: 'Alegreya Sans', Arial, sans-serif !important; font-weight: 800 !important; color: {ENCRE}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.stApp::before {{ content: ""; position: fixed; top: 0; left: 0; right: 0; height: 8px; z-index: 999999;
  background: linear-gradient(90deg, {CORAIL} 0 33.3%, {SARCELLE} 33.3% 66.6%, {AZUR} 66.6% 100%); }}
div[data-testid="stMetricValue"] {{ color: #D4532F; font-weight: 600; }}
</style>""", unsafe_allow_html=True)


def eur(v, signe=False):
    s = f"{abs(v):,.0f}".replace(",", " ")
    if signe:
        return f"{'+' if v >= 0 else '−'}{s} €"
    return f"{'−' if v < 0 else ''}{s} €"


@st.cache_data
def charger_donnees():
    df, journal = P.nettoyer(ICI / "data" / "riviera_du_levant.csv")
    return df, journal


@st.cache_resource
def charger_modeles():
    """Charge les modèles pré-entraînés ; s'ils sont absents ou incompatibles
    (autre version de scikit-learn ou de xgboost), les ré-entraîne en mémoire (quelques secondes)."""
    df, _ = charger_donnees()
    try:
        m = joblib.load(ART / "modeles.joblib")
        xgb, ridge = m["xgb"], m["ridge"]
        xgb.predict(P.profil_vers_X({"bedrooms": 1, "guests": 2, "baths": 1}))
        return xgb, ridge, "pré-entraînés"
    except Exception:
        xgb, ridge = P.entrainer(df)
        return xgb, ridge, "ré-entraînés au lancement"


@st.cache_data
def charger_metriques():
    return json.loads((ART / "metriques.json").read_text(encoding="utf-8"))


df, journal = charger_donnees()
xgb, ridge, origine = charger_modeles()
M = charger_metriques()
COEFS, INTERCEPT, ECHELLE = P.coefficients_ridge(ridge)
RES = pd.DataFrame(M["resultats_cv"])
IMB = M["imbrique"]
MAE_MODELE = IMB["xgb_mae"]   # erreur mesurée par validation croisée imbriquée (sans biais de réglage)

st.sidebar.title("Airbnb Market Intelligence")
st.sidebar.caption("Riviera du Levant, Guadeloupe · Le Gosier, Sainte-Anne, Saint-François")
page = st.sidebar.radio("Navigation", ["1 · Le projet", "2 · Market Explorer",
                                       "3 · Modèles", "4 · Property Simulator", "5 · Limites"])
st.sidebar.divider()
st.sidebar.caption(f"{len(df):,} annonces · source AirROI · modèles {origine}".replace(",", " "))
st.sidebar.caption("Marvin Jernidier · MJ Metrix")

# ------------------------------------------------------------------ 1. PROJET
if page.startswith("1"):
    st.title("Airbnb Market Intelligence")
    st.subheader("Estimer le revenu annuel d'une location courte durée à partir de ses caractéristiques")
    c = st.columns(4)
    c[0].metric("Annonces analysées", f"{len(df):,}".replace(",", " "), f"sur {journal[0][1]:,} brutes".replace(",", " "), delta_color="off")
    c[1].metric("Revenu annuel médian", eur(df[P.CIBLE].median()))
    c[2].metric("R² (validation croisée imbriquée)", f"{IMB['xgb_r2']:.3f}".replace(".", ","),
                f"{IMB['gain_r2']:+.3f} vs première version".replace(".", ","))
    c[3].metric("Erreur absolue moyenne", eur(MAE_MODELE), f"{(MAE_MODELE / M['baseline_mae'] - 1) * 100:.0f} % vs estimation naïve", delta_color="inverse")
    st.markdown("""
**Trois questions métier**
1. Quelles caractéristiques d'un logement sont associées aux meilleurs revenus ?
2. Quels équipements présentent le plus grand intérêt économique ?
3. Quel revenu annuel peut-on attendre d'un logement d'un profil donné ?

**Démarche :** données AirROI → nettoyage traçable → trois familles de modèles comparées en validation croisée → analyse des erreurs → modèle amélioré (studios recodés, mots-clés du titre, XGBoost) → estimation et explication.
""")
    st.markdown("**Du jeu brut au jeu modélisable**")
    j = pd.DataFrame(journal, columns=["Étape", "Annonces restantes"])
    j["Retirées"] = (-j["Annonces restantes"].diff()).fillna(0).astype(int)
    st.dataframe(j, hide_index=True)

# ------------------------------------------------------------------ 2. EXPLORER
elif page.startswith("2"):
    st.title("Market Explorer")
    f1, f2, f3 = st.columns([2, 2, 3])
    communes = f1.multiselect("Communes", P.COMMUNES, default=P.COMMUNES)
    ch = f2.slider("Chambres (0 = studio)", 0, 7, (0, 7))
    equip = f3.multiselect("Équipements requis", P.EQUIPEMENTS, format_func=P.LIBELLES.get)
    d = df[df.locality.isin(communes)]
    d = d[d.bedrooms.between(*ch)]
    for e in equip:
        d = d[d[e] == 1]
    if d.empty:
        st.warning("Aucune annonce ne correspond à ces filtres.")
        st.stop()
    k = st.columns(4)
    k[0].metric("Annonces", len(d))
    k[1].metric("Revenu médian", eur(d[P.CIBLE].median()), eur(d[P.CIBLE].median() - df[P.CIBLE].median(), True) + " vs marché")
    k[2].metric("Revenu moyen", eur(d[P.CIBLE].mean()))
    k[3].metric("Part de Superhosts", f"{d.superhost.mean() * 100:.0f} %")

    g1, g2 = st.columns(2)
    g1.markdown("**Distribution du revenu annuel**")
    g1.altair_chart(alt.Chart(d).mark_bar(color=BLEU).encode(
        x=alt.X(f"{P.CIBLE}:Q", bin=alt.Bin(maxbins=35), title="Revenu sur 12 mois (€)"),
        y=alt.Y("count()", title="Annonces"), tooltip=["count()"]))
    g2.markdown("**Revenu médian selon le nombre de chambres**")
    t = d.groupby("bedrooms")[P.CIBLE].agg(["median", "count"]).reset_index()
    t["bedrooms"] = t["bedrooms"].astype(int)
    g2.altair_chart(alt.Chart(t).mark_bar(color=BLEU).encode(
        x=alt.X("bedrooms:O", title="Chambres"), y=alt.Y("median:Q", title="Revenu médian (€)"),
        tooltip=[alt.Tooltip("median:Q", format=",.0f", title="Médiane"), alt.Tooltip("count:Q", title="Annonces")]))

    st.markdown("**Écart brut de revenu moyen par équipement** (avec − sans, sur la sélection)")
    lignes = []
    for e in P.EQUIPEMENTS:
        a, s = d[d[e] == 1][P.CIBLE], d[d[e] == 0][P.CIBLE]
        if len(a) >= 10 and len(s) >= 10:
            lignes.append({"Équipement": P.LIBELLES[e], "Écart brut": a.mean() - s.mean(), "Annonces équipées": len(a)})
    if lignes:
        st.altair_chart(alt.Chart(pd.DataFrame(lignes)).mark_bar().encode(
            x=alt.X("Écart brut:Q", title="€ / an"), y=alt.Y("Équipement:N", sort="-x", title=None),
            color=alt.condition(alt.datum["Écart brut"] > 0, alt.value(BLEU), alt.value(ORANGE)),
            tooltip=[alt.Tooltip("Écart brut:Q", format="+,.0f"), "Annonces équipées"]))
    st.caption("Comparaison brute : elle ne contrôle pas la taille du logement. Un logement avec lave-vaisselle "
               "a en moyenne presque une chambre de plus. L'effet ajusté est dans l'onglet Modèles.")

    st.markdown("**Annonces les plus performantes de la sélection**")
    cols = ["listing_name", "locality", P.CIBLE, "bedrooms", "baths", "guests", "superhost", "num_reviews"]
    st.dataframe(d.sort_values(P.CIBLE, ascending=False)[cols].head(10).rename(columns={
        "listing_name": "Annonce", "locality": "Commune", P.CIBLE: "Revenu annuel (€)", "bedrooms": "Chambres",
        "baths": "Sdb", "guests": "Voyageurs", "superhost": "Superhost", "num_reviews": "Avis"}),
        hide_index=True)

# ------------------------------------------------------------------ 3. MODELES
elif page.startswith("3"):
    st.title("Modèles : comparaison et interprétation")
    st.markdown("**Première version** : trois familles de modèles sur 16 variables. **Modèle amélioré** : studios recodés à "
                "0 chambre, 12 mots-clés extraits du titre de l'annonce, XGBoost. Hyperparamètres réglés par recherche "
                "aléatoire, évaluation par **validation croisée à 5 plis**.")
    t = RES.copy()
    t["R² (CV)"] = t.apply(lambda r: f"{r.r2:.3f} ± {r.r2_std:.3f}", axis=1)
    t["MAE (CV)"] = t.mae.map(eur)
    t["R² entraînement"] = t.r2_train.map(lambda v: f"{v:.3f}")
    st.dataframe(t[["modele", "variables", "R² (CV)", "MAE (CV)", "R² entraînement"]].rename(
        columns={"modele": "Modèle", "variables": "Variables"}), hide_index=True)
    k = st.columns(3)
    k[0].metric("Mesure sans biais : première version", f"R² {IMB['ref_r2']:.3f}".replace(".", ","), eur(IMB["ref_mae"]) + " d'erreur", delta_color="off")
    k[1].metric("Mesure sans biais : modèle amélioré", f"R² {IMB['xgb_r2']:.3f}".replace(".", ","), eur(IMB["xgb_mae"]) + " d'erreur", delta_color="off")
    k[2].metric("Gain de R²", f"{IMB['gain_r2']:+.3f}".replace(".", ","), f"positif sur {IMB['plis_gagnes']} mesures sur {IMB['plis']}", delta_color="off")
    st.info("Le tableau note chaque modèle sur les plis qui ont servi à régler ses hyperparamètres : ces scores sont légèrement "
            "optimistes. La **validation croisée imbriquée** refait le réglage à l'intérieur de chaque pli et donne la mesure "
            f"retenue. Référence naïve (prédire la moyenne) : MAE {eur(M['baseline_mae'])}.")

    a, b = st.columns(2)
    a.markdown("**XGBoost : sur quoi repose la précision ?**  \nPermutation importance (perte de R² sur les plis de validation), 14 premières variables")
    pi = pd.DataFrame({"Variable": [P.LIBELLES[f] for f in P.FEATURES],
                       "Perte de R²": [M["permutation"][f] for f in P.FEATURES]}).nlargest(14, "Perte de R²")
    a.altair_chart(alt.Chart(pi).mark_bar(color=BLEU).encode(
        x="Perte de R²:Q", y=alt.Y("Variable:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("Perte de R²:Q", format=".3f")]))
    b.markdown("**Ridge : effet estimé en euros, toutes choses égales**  \nVariables continues : par unité indiquée")
    unites = {f: f" (+{ECHELLE[f]['iqr']:g})".replace(".", ",") for f in P.CONTINUES}
    co = pd.DataFrame({"Variable": [P.LIBELLES[f] + unites.get(f, "") for f in P.FEATURES],
                       "€ / an": [COEFS[f] for f in P.FEATURES]})
    b.altair_chart(alt.Chart(co).mark_bar().encode(
        x="€ / an:Q", y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        color=alt.condition(alt.datum["€ / an"] > 0, alt.value(BLEU), alt.value(ORANGE)),
        tooltip=[alt.Tooltip("€ / an:Q", format="+,.0f")]), height=620)
    st.caption("Deux questions différentes. La permutation importance mesure ce dont le modèle a besoin pour bien prédire. "
               "Les coefficients Ridge mesurent une association en euros à autres variables fixées. Aucune des deux n'est une preuve de causalité. "
               "Les variables « Titre » valent 1 quand le titre de l'annonce contient le mot-clé.")

    st.markdown("**XGBoost : prédictions hors échantillon vs revenus réels**")
    hp = pd.read_csv(ART / "predictions_hors_pli.csv")
    pts = alt.Chart(hp).mark_circle(size=18, opacity=0.4, color=BLEU).encode(
        x=alt.X("reel:Q", title="Revenu réel (€)"), y=alt.Y("predit:Q", title="Revenu prédit (€)"))
    diag = alt.Chart(pd.DataFrame({"v": [0, 85000]})).mark_line(color=GRIS, strokeDash=[5, 4]).encode(x="v:Q", y="v:Q")
    st.altair_chart(pts + diag)
    st.caption("Le modèle surestime les petits revenus et sous-estime les plus élevés (régression vers la moyenne).")

# ------------------------------------------------------------------ 4. SIMULATEUR
elif page.startswith("4"):
    st.title("Property Simulator")
    st.caption("Estimation par XGBoost · explication par le modèle Ridge · comparaison avec les annonces réelles")
    g, dr = st.columns([1, 1.25])
    with g:
        st.markdown("**Le logement**")
        k1, k2 = st.columns(2)
        commune = k1.selectbox("Commune", P.COMMUNES)
        type_de_bien = k2.selectbox("Type de bien", list(P.TYPES_DE_BIEN))
        studio = type_de_bien == "Studio"
        c1, c2, c3 = st.columns(3)
        bedrooms = 0 if studio else c1.number_input("Chambres", 1, 6, 1)
        if studio:
            c1.markdown("Chambres  \n**0 (studio)**")
        baths = c2.number_input("Salles de bain", 1.0, 5.0, 1.0, 0.5)
        guests = c3.number_input("Voyageurs", 1, 14, 3)
        st.markdown("**Équipements**")
        defaut = {"air_conditioning": True, "cable_tv": True}
        cols = st.columns(3)
        eq = {e: cols[i % 3].checkbox(P.LIBELLES[e], value=defaut.get(e, False)) for i, e in enumerate(P.EQUIPEMENTS)}
        st.markdown("**Ce que l'annonce met en avant dans son titre**")
        cols = st.columns(3)
        mis = {t: cols[i % 3].checkbox(P.LIBELLES[t].replace("Titre : ", "").capitalize(), key=t) for i, t in enumerate(P.MISES_EN_AVANT)}
        st.markdown("**Hypothèses d'exploitation**")
        superhost = st.toggle("Statut Superhost", value=True)
        num_reviews = st.slider("Nombre d'avis", 5, 300, 65)
        st.caption("Ces deux variables n'existent qu'après mise en location. Pour un bien jamais exploité, "
                   "ce sont des hypothèses, pas des caractéristiques.")

    profil = {"commune": commune, "type_de_bien": type_de_bien, "bedrooms": bedrooms, "baths": baths, "guests": guests,
              "superhost": int(superhost), "num_reviews": num_reviews, **{e: int(v) for e, v in eq.items()},
              **{t: int(v) for t, v in mis.items()}}
    Xp = P.profil_vers_X(profil)
    est_xgb, est_ridge = float(xgb.predict(Xp)[0]), float(ridge.predict(Xp)[0])

    with dr:
        m1, m2 = st.columns(2)
        m1.metric("Revenu annuel estimé (XGBoost)", eur(est_xgb))
        m2.metric("Estimation du modèle Ridge", eur(est_ridge))
        st.caption(f"Erreur absolue moyenne du modèle en validation croisée imbriquée : ± {eur(MAE_MODELE)}. "
                   f"Ce n'est pas un intervalle de confiance, mais l'ordre de grandeur de l'erreur typique.")

        comp = df[(df.locality == commune) & (df.bedrooms == bedrooms)]
        libelle = "studios" if studio else f"annonces de {bedrooms} chambre(s)"
        if len(comp) >= 15:
            q1, med, q3 = comp[P.CIBLE].quantile([.25, .5, .75])
            st.markdown(f"**Comparables réels** : {len(comp)} {libelle} à {commune}. "
                        f"Médiane {eur(med)}, moitié centrale entre {eur(q1)} et {eur(q3)}.")
        else:
            st.markdown(f"**Comparables réels** : seulement {len(comp)} {libelle} à {commune}. "
                        f"Estimation à lire avec prudence.")

        st.markdown("**Ce qui explique l'estimation Ridge** (écart au logement de base, en € / an)")
        contrib = []
        for f in P.FEATURES:
            v = float(Xp[f].iloc[0])
            if f in P.CONTINUES:
                v = (v - ECHELLE[f]["centre"]) / ECHELLE[f]["iqr"]
            if abs(COEFS[f] * v) > 1:
                contrib.append({"Variable": P.LIBELLES[f], "Contribution": COEFS[f] * v})
        if contrib:
            st.altair_chart(alt.Chart(pd.DataFrame(contrib)).mark_bar().encode(
                x=alt.X("Contribution:Q", title="€ / an"), y=alt.Y("Variable:N", sort="-x", title=None),
                color=alt.condition(alt.datum.Contribution > 0, alt.value(BLEU), alt.value(ORANGE)),
                tooltip=[alt.Tooltip("Contribution:Q", format="+,.0f")]))
        st.caption(f"Base du modèle Ridge : {eur(INTERCEPT)} (logement médian de {ECHELLE['bedrooms']['centre']:g} chambre, "
                   f"{ECHELLE['baths']['centre']:g} salle de bain, {ECHELLE['guests']['centre']:g} voyageurs, "
                   f"{ECHELLE['num_reviews']['centre']:g} avis, sans équipement ni mot-clé, au Gosier).")

    st.divider()
    st.markdown("**Et si j'ajoutais un équipement ?** Variation de l'estimation, les autres caractéristiques restant identiques.")
    lignes = []
    for e in P.EQUIPEMENTS:
        if not eq[e]:
            x2 = P.profil_vers_X({**profil, e: 1})
            lignes.append({"Équipement ajouté": P.LIBELLES[e],
                           "XGBoost": eur(float(xgb.predict(x2)[0]) - est_xgb, True),
                           "Ridge": eur(float(ridge.predict(x2)[0]) - est_ridge, True)})
    if lignes:
        st.dataframe(pd.DataFrame(lignes), hide_index=True)
    st.warning("Ces écarts sont des **associations statistiques** observées sur des annonces existantes, pas des effets "
               "causaux garantis. Ils orientent une décision ; ils ne remplacent pas un calcul de rentabilité avec les coûts.")

# ------------------------------------------------------------------ 5. LIMITES
else:
    st.title("Limites et perspectives")
    a, b = st.columns(2)
    a.markdown("""
**Limites**
- **Données observationnelles** : associations, pas causalité.
- **Biais de sélection** : logements entiers avec au moins 110 nuits réservées sur 12 mois. Le modèle décrit des biens déjà actifs.
- **Variables absentes** : prix par nuit, saisonnalité, qualité des photos, notes, localisation fine.
- **Mots-clés du titre** : extraction simple par expressions régulières, sur des titres parfois traduits ou peu soignés.
- **Domaine de validité** : revenus entre 2 000 € et 82 495 €, trois communes.
- **Précision** : R² de 0,60 ; les revenus les plus élevés restent sous-estimés.
- **Revenu, pas rentabilité** : aucun coût d'acquisition ni d'exploitation.
""")
    b.markdown("""
**Perspectives**
- Enrichir avec les variables AirROI non exportées (type de bien, notes, coordonnées GPS, nombre de lits).
- Analyser les descriptions et les avis par text mining, au-delà du titre.
- Géolocalisation fine : distance à la plage, quartiers.
- Données temporelles : saisonnalité, performances à 90 jours.
- Sélection de comparables par proximité et similarité.
- Modéliser les coûts pour passer du revenu au rendement.
- Intervalles de prédiction (régression quantile).
""")
