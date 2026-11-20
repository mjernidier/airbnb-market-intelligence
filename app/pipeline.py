"""Pipeline commun : nettoyage, variables et entraînement, identiques au notebook du projet
(section « Modèle amélioré » : studios recodés à 0 chambre, mots-clés du titre, XGBoost).

Utilisé par preparer_modeles.py (une fois, avant la soutenance) et par app.py
(secours : ré-entraînement en mémoire si les artefacts sont absents ou incompatibles).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

CIBLE = "ttm_revenue"
CONTINUES = ["bedrooms", "guests", "baths", "num_reviews"]
EQUIPEMENTS = ["pool", "air_conditioning", "dishwasher", "dedicated_workspace",
               "waterfront", "hot_tub", "bbq_grill", "cable_tv", "gym"]
COMMUNES = ["Le Gosier", "Sainte-Anne", "Saint-François"]  # Le Gosier = référence
# Text mining : mots-clés du titre de l'annonce (liste fixée sans regarder le revenu)
MOTS_CLES = {
    "t_villa": r"villa",
    "t_studio": r"studio",
    "t_appart": r"apartment|appartement|\bt[1-4]\b|\bf[1-4]\b|flat|duplex",
    "t_bungalow": r"bungalow|cottage|\bkaz\b|\bcase\b|gite|gîte|lodge",
    "t_house": r"house|maison",
    "t_sea": r"sea ?view|vue mer|ocean|lagoon|lagon|panoram",
    "t_beach": r"beach|plage|beachfront|waterfront|feet in the water|pieds dans",
    "t_luxe": r"luxury|luxe|standing|prestige|exceptional|premium|chic|architect|haut de gamme|5 ?\*|star",
    "t_pool": r"pool|piscine",
    "t_private": r"private|privée|privative",
    "t_marina": r"marina|golf",
    "t_jacuzzi": r"jacuzzi|spa|hot tub",
}
TITRE = list(MOTS_CLES)
TYPES_DE_BIEN = {"Appartement": "t_appart", "Villa": "t_villa", "Maison": "t_house",
                 "Bungalow, gîte": "t_bungalow", "Studio": "t_studio", "Autre": None}
MISES_EN_AVANT = ["t_sea", "t_beach", "t_luxe", "t_pool", "t_private", "t_marina", "t_jacuzzi"]

FEATURES_BASE = (["bedrooms", "guests", "baths", "superhost", "num_reviews"] + EQUIPEMENTS
                 + ["locality_Saint-François", "locality_Sainte-Anne"])   # les 16 variables de référence
FEATURES = FEATURES_BASE + TITRE                                           # 28 variables du modèle amélioré
BINAIRES = [c for c in FEATURES if c not in CONTINUES]

LIBELLES = {
    "bedrooms": "Chambres", "guests": "Voyageurs", "baths": "Salles de bain",
    "superhost": "Superhost", "num_reviews": "Nombre d'avis", "pool": "Piscine",
    "air_conditioning": "Climatisation", "dishwasher": "Lave-vaisselle",
    "dedicated_workspace": "Espace de travail", "waterfront": "Front de mer",
    "hot_tub": "Jacuzzi", "bbq_grill": "Barbecue", "cable_tv": "TV câblée",
    "gym": "Salle de sport", "locality_Saint-François": "Saint-François (vs Gosier)",
    "locality_Sainte-Anne": "Sainte-Anne (vs Gosier)",
    "t_villa": "Titre : villa", "t_studio": "Titre : studio", "t_appart": "Titre : appartement",
    "t_bungalow": "Titre : bungalow, gîte", "t_house": "Titre : maison", "t_sea": "Titre : vue mer",
    "t_beach": "Titre : plage", "t_luxe": "Titre : haut de gamme", "t_pool": "Titre : piscine",
    "t_private": "Titre : privé", "t_marina": "Titre : marina, golf", "t_jacuzzi": "Titre : jacuzzi, spa",
}

# Hyperparamètres issus du RandomizedSearchCV du notebook (20 essais, CV 5 plis, random_state=42)
XGB_PARAMS = dict(n_estimators=393, learning_rate=0.022004527434741072, max_depth=4, min_child_weight=3,
                  subsample=0.6185801650879991, colsample_bytree=0.7159725093210578,
                  reg_lambda=4.446862319918233, random_state=42, n_jobs=1)
RF_TITRE_PARAMS = dict(n_estimators=188, max_depth=11, min_samples_leaf=1, max_features=0.3777641034224495,
                       random_state=42, n_jobs=-1)
# Modèles de référence (16 variables, première version du notebook)
RF_REF_PARAMS = dict(n_estimators=207, max_depth=13, min_samples_leaf=3, random_state=42, n_jobs=-1)
RIDGE_ALPHA = 10.626955533913808  # alpha retenu par la recherche aléatoire du notebook

SEUIL_REVENU, SEUIL_AVIS, FACTEUR_IQR = 2000, 5, 3.0


def _id_valide(v):
    try:
        int(v)
        return True
    except (ValueError, TypeError):
        return False


def nettoyer(chemin_csv):
    """Reproduit les 4 étapes de nettoyage du notebook. Retourne (df_propre, journal)."""
    brut = pd.read_csv(chemin_csv, encoding="utf-8-sig")
    journal = [("Export brut AirROI", len(brut))]
    df = brut[brut["listing_id"].apply(_id_valide)].copy()
    journal.append(("Lignes corrompues retirées", len(df)))
    df[CIBLE] = pd.to_numeric(df[CIBLE], errors="coerce")
    df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce")
    df = df.dropna(subset=[CIBLE, "num_reviews"]).copy()
    journal.append(("Revenu ou avis manquant retirés", len(df)))
    df = df[~((df[CIBLE] < SEUIL_REVENU) | (df["num_reviews"] < SEUIL_AVIS))].copy()
    journal.append(("Inactives ou trop récentes retirées", len(df)))
    q1, q3 = df[CIBLE].quantile(0.25), df[CIBLE].quantile(0.75)
    borne = q3 + FACTEUR_IQR * (q3 - q1)
    df = df[df[CIBLE] <= borne].reset_index(drop=True)
    journal.append((f"Valeurs extrêmes retirées (> {borne:,.0f} €)".replace(",", " "), len(df)))
    for c in ["bedrooms", "guests", "baths", "superhost"] + EQUIPEMENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return enrichir(df), journal


def enrichir(df):
    """Variables du modèle amélioré. Les annonces sans nombre de chambres sont des studios
    (la source ne code jamais 0 chambre) : on les recode à 0 au lieu de les imputer."""
    df = df.copy()
    df["bedrooms"] = df["bedrooms"].fillna(0)
    titres = df["listing_name"].astype(str).str.lower()
    for nom, motif in MOTS_CLES.items():
        df[nom] = titres.str.contains(motif, regex=True).astype(int)
    return df


def matrice(df):
    X = df.copy()
    X["locality_Saint-François"] = (X["locality"] == "Saint-François").astype(int)
    X["locality_Sainte-Anne"] = (X["locality"] == "Sainte-Anne").astype(int)
    return X[FEATURES], X[CIBLE]


def preprocesseur():
    return ColumnTransformer([
        ("continues", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                ("scaler", RobustScaler())]), CONTINUES),
        ("binaires", SimpleImputer(strategy="median"), BINAIRES),
    ])


def entrainer(df):
    """XGBoost (estimation) et Ridge (explication), ajustés sur toutes les annonces nettoyées,
    avec les 28 variables du modèle amélioré."""
    X, y = matrice(df)
    xgb = Pipeline([("preprocess", preprocesseur()), ("model", XGBRegressor(**XGB_PARAMS))]).fit(X, y)
    ridge = Pipeline([("preprocess", preprocesseur()), ("model", Ridge(alpha=RIDGE_ALPHA))]).fit(X, y)
    return xgb, ridge


def coefficients_ridge(ridge):
    """Coefficients en euros, remis dans l'ordre de FEATURES, + paramètres du RobustScaler."""
    noms = CONTINUES + BINAIRES
    coefs = dict(zip(noms, ridge.named_steps["model"].coef_))
    scaler = ridge.named_steps["preprocess"].named_transformers_["continues"].named_steps["scaler"]
    echelle = {c: {"centre": float(m), "iqr": float(s)}
               for c, m, s in zip(CONTINUES, scaler.center_, scaler.scale_)}
    return {f: float(coefs[f]) for f in FEATURES}, float(ridge.named_steps["model"].intercept_), echelle


def profil_vers_X(profil):
    ligne = {f: 0.0 for f in FEATURES}
    for k, v in profil.items():
        if k in ligne:
            ligne[k] = float(v)
    t = TYPES_DE_BIEN.get(profil.get("type_de_bien", "Autre"))
    if t:
        ligne[t] = 1.0
    commune = profil.get("commune", "Le Gosier")
    ligne["locality_Saint-François"] = 1.0 if commune == "Saint-François" else 0.0
    ligne["locality_Sainte-Anne"] = 1.0 if commune == "Sainte-Anne" else 0.0
    return pd.DataFrame([ligne])[FEATURES]
