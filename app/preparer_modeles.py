"""À lancer UNE fois avant la soutenance :  python preparer_modeles.py   (2 à 3 minutes)
Nettoie les données, évalue les modèles en validation croisée (simple puis imbriquée),
sauvegarde modèles et métriques. Reproduit les chiffres du notebook."""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, RepeatedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

import pipeline as P

warnings.filterwarnings("ignore")
ICI = Path(__file__).parent
(ICI / "artefacts").mkdir(exist_ok=True)
df, journal = P.nettoyer(ICI / "data" / "riviera_du_levant.csv")
df.to_csv(ICI / "artefacts" / "riviera_du_levant_cleaned.csv", index=False, encoding="utf-8-sig")
X, y = P.matrice(df)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Jeu de référence : 16 variables, chambres manquantes imputées par la médiane (première version)
X_ref = X[P.FEATURES_BASE].copy()
X_ref["bedrooms"] = X_ref["bedrooms"].replace(0, np.nan)
BIN_REF = [c for c in P.FEATURES_BASE if c not in P.CONTINUES]


def pre_ref():
    return ColumnTransformer([
        ("continues", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), P.CONTINUES),
        ("binaires", SimpleImputer(strategy="median"), BIN_REF)])


def pipe(pre, modele):
    return Pipeline([("preprocess", pre), ("model", modele)])


candidats = [
    ("XGBoost + titre", "28 (modèle amélioré)", pipe(P.preprocesseur(), XGBRegressor(**P.XGB_PARAMS)), X),
    ("Random Forest + titre", "28 (modèle amélioré)", pipe(P.preprocesseur(), RandomForestRegressor(**P.RF_TITRE_PARAMS)), X),
    ("Ridge + titre", "28 (modèle amélioré)", pipe(P.preprocesseur(), Ridge(alpha=P.RIDGE_ALPHA)), X),
    ("Random Forest", "16 (référence)", pipe(pre_ref(), RandomForestRegressor(**P.RF_REF_PARAMS)), X_ref),
    ("HistGradientBoosting", "16 (référence)", pipe(pre_ref(), HistGradientBoostingRegressor(
        max_iter=70, learning_rate=0.06454749016213018, max_leaf_nodes=42, random_state=42)), X_ref),
    ("Ridge", "16 (référence)", pipe(pre_ref(), Ridge(alpha=P.RIDGE_ALPHA)), X_ref),
]
resultats = []
for nom, variables, m, Xm in candidats:
    s = cross_validate(m, Xm, y, cv=cv, scoring=["r2", "neg_mean_absolute_error"], return_train_score=True)
    resultats.append({"modele": nom, "variables": variables, "r2": s["test_r2"].mean(), "r2_std": s["test_r2"].std(),
                      "mae": -s["test_neg_mean_absolute_error"].mean(), "r2_train": s["train_r2"].mean()})
s = cross_validate(DummyRegressor(strategy="mean"), X, y, cv=cv, scoring=["neg_mean_absolute_error"])
baseline_mae = -s["test_neg_mean_absolute_error"].mean()

# Validation croisée imbriquée : le réglage est refait dans chaque pli externe (mesure sans biais)
cv_externe = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
espace_xgb = {"model__n_estimators": randint(150, 700), "model__learning_rate": loguniform(0.01, 0.15),
              "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4),
              "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 10),
              "model__reg_lambda": loguniform(0.5, 20)}
espace_rf = {"model__n_estimators": randint(100, 250), "model__max_depth": randint(3, 15),
             "model__min_samples_leaf": randint(1, 10)}


def imbriquee(p, espace, Xm, n_iter):
    r2s, maes = [], []
    for i, (tr, te) in enumerate(cv_externe.split(Xm)):
        rs = RandomizedSearchCV(clone(p), espace, n_iter=n_iter, cv=KFold(5, shuffle=True, random_state=100 + i),
                                scoring="r2", random_state=42, n_jobs=-1).fit(Xm.iloc[tr], y.iloc[tr])
        pr = rs.predict(Xm.iloc[te])
        r2s.append(r2_score(y.iloc[te], pr))
        maes.append(mean_absolute_error(y.iloc[te], pr))
    return np.array(r2s), np.array(maes)


r2_ref, mae_ref = imbriquee(pipe(pre_ref(), RandomForestRegressor(random_state=42, n_jobs=1)), espace_rf, X_ref, 15)
r2_xgb, mae_xgb = imbriquee(pipe(P.preprocesseur(), XGBRegressor(random_state=42, n_jobs=1)), espace_xgb, X, 20)
imbrique = {"ref_r2": r2_ref.mean(), "ref_r2_std": r2_ref.std(), "ref_mae": mae_ref.mean(),
            "xgb_r2": r2_xgb.mean(), "xgb_r2_std": r2_xgb.std(), "xgb_mae": mae_xgb.mean(),
            "gain_r2": (r2_xgb - r2_ref).mean(), "gain_r2_std": (r2_xgb - r2_ref).std(),
            "plis_gagnes": int((r2_xgb > r2_ref).sum()), "plis": len(r2_ref)}

xgb, ridge = P.entrainer(df)
pipe_xgb = pipe(P.preprocesseur(), XGBRegressor(**P.XGB_PARAMS))
hors_pli = cross_val_predict(pipe_xgb, X, y, cv=cv)
hors_pli_ref = cross_val_predict(candidats[3][2], X_ref, y, cv=cv)
imps = []
for tr, va in cv.split(X):
    p = clone(pipe_xgb).fit(X.iloc[tr], y.iloc[tr])
    imps.append(permutation_importance(p, X.iloc[va], y.iloc[va], n_repeats=30, random_state=42,
                                       scoring="r2").importances_mean)
coefs, intercept, echelle = P.coefficients_ridge(ridge)

# Variante « bien jamais loué » : sans nombre d'avis ni statut Superhost
f_neuf = [f for f in P.FEATURES if f not in ("num_reviews", "superhost")]
pre_neuf = ColumnTransformer([
    ("continues", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]),
     [c for c in P.CONTINUES if c != "num_reviews"]),
    ("binaires", SimpleImputer(strategy="median"), [c for c in P.BINAIRES if c != "superhost"])])
s = cross_validate(pipe(pre_neuf, XGBRegressor(**P.XGB_PARAMS)), X[f_neuf], y, cv=cv,
                   scoring=["r2", "neg_mean_absolute_error"])
neuf = {"r2": s["test_r2"].mean(), "r2_std": s["test_r2"].std(), "mae": -s["test_neg_mean_absolute_error"].mean()}

# Erreurs par quintile de revenu réel
e = pd.DataFrame({"reel": y, "xgb": hors_pli - y, "ref": hors_pli_ref - y})
e["q"] = pd.qcut(e.reel, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
quintiles = [{"quintile": q, "reel_moyen": g.reel.mean(), "biais_xgb": g.xgb.mean(), "mae_xgb": g.xgb.abs().mean(),
              "biais_ref": g.ref.mean(), "mae_ref": g.ref.abs().mean()} for q, g in e.groupby("q", observed=True)]

metriques = {
    "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "journal_nettoyage": journal,
    "resultats_cv": resultats, "baseline_mae": baseline_mae, "imbrique": imbrique, "bien_neuf": neuf,
    "quintiles": quintiles,
    "permutation": dict(zip(P.FEATURES, np.mean(imps, axis=0).tolist())),
    "permutation_std": dict(zip(P.FEATURES, np.std(imps, axis=0).tolist())),
    "ridge_coefs": coefs, "ridge_intercept": intercept, "scaler": echelle,
    "mots_cles_effectifs": {t: int(df[t].sum()) for t in P.TITRE},
}
(ICI / "artefacts" / "metriques.json").write_text(json.dumps(metriques, ensure_ascii=False, indent=1), encoding="utf-8")
pd.DataFrame({"reel": y, "predit": hors_pli}).to_csv(ICI / "artefacts" / "predictions_hors_pli.csv", index=False)
joblib.dump({"xgb": xgb, "ridge": ridge, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
            ICI / "artefacts" / "modeles.joblib")
print("Nettoyage :", journal)
for r in resultats:
    print(f"{r['modele']:22s} {r['variables']:22s} R² {r['r2']:.3f} ± {r['r2_std']:.3f} | MAE {r['mae']:,.0f} € | R² train {r['r2_train']:.3f}")
print(f"Baseline (moyenne) MAE {baseline_mae:,.0f} €")
print(f"Imbriquée : référence R² {imbrique['ref_r2']:.3f} MAE {imbrique['ref_mae']:,.0f} € | "
      f"XGBoost + titre R² {imbrique['xgb_r2']:.3f} MAE {imbrique['xgb_mae']:,.0f} € | "
      f"gain {imbrique['gain_r2']:+.3f} sur {imbrique['plis_gagnes']}/{imbrique['plis']} plis")
print(f"Bien jamais loué : R² {neuf['r2']:.3f} | MAE {neuf['mae']:,.0f} €")
print("Artefacts sauvegardés dans", ICI / "artefacts")
