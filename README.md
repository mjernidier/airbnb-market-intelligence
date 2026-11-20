# Airbnb Market Intelligence

Estimation du revenu annuel d'une location courte durée sur la Riviera du Levant (Le Gosier, Sainte-Anne, Saint-François), en Guadeloupe. Projet de fin de formation Data Scientist (DataScientest), réalisé par Marvin Jernidier.

## Objectif

Un investisseur ou un propriétaire se pose trois questions avant de se lancer : quelles caractéristiques d'un logement sont associées aux meilleurs revenus, quels équipements méritent l'investissement, et combien peut rapporter un bien d'un profil donné. Le projet y répond à partir de données de marché, avec un modèle de régression et une application de démonstration.

Le modèle prédit un revenu annuel, pas une rentabilité : les coûts d'acquisition et d'exploitation ne figurent pas dans les données.

## Données

Les données proviennent de l'API AirROI (recherche d'annonces par marché), collectées commune par commune avec deux filtres : logement entier, et au moins 110 nuits réservées sur les douze derniers mois. L'export contient 1 220 annonces et 20 colonnes. Après nettoyage, 1 163 annonces sont conservées :

| Étape | Annonces restantes |
|---|---|
| Export brut | 1 220 |
| Lignes corrompues retirées (12) | 1 208 |
| Revenu ou nombre d'avis manquant (10) | 1 198 |
| Moins de 5 avis (10) | 1 188 |
| Revenu supérieur à 82 495 €, soit Q3 + 3 × IQR (25) | 1 163 |

La variable cible est `ttm_revenue`, le revenu sur douze mois glissants. Revenu moyen 23 683 €, médian 19 286 €.

## Démarche

1. Analyse exploratoire : distribution des revenus, écarts bruts par équipement et par commune.
2. Première version : 16 variables (taille, équipements, hôte, commune), prétraitement dans un pipeline scikit-learn (imputation, RobustScaler, encodage), trois familles de modèles comparées en validation croisée à 5 plis avec recherche aléatoire d'hyperparamètres. Ridge, Random Forest et HistGradientBoosting plafonnent au même niveau, autour de 0,58 de R².
3. Pistes testées et écartées : ratios de standing, variables dérivées, segmentation KMeans, cible en logarithme. Aucune ne dépasse la variabilité entre plis.
4. Modèle amélioré : les 101 annonces sans nombre de chambres sont des studios et sont recodées à 0 chambre ; 12 mots-clés sont extraits du titre des annonces (villa, studio, vue mer, plage, luxe, etc.) ; un modèle XGBoost est réglé sur ces 28 variables.
5. Mesure sans biais par validation croisée imbriquée (5 plis répétés 2 fois, réglage refait dans chaque pli d'entraînement).

## Résultats

| Modèle | R² | Erreur absolue moyenne |
|---|---|---|
| Référence naïve (prédire la moyenne) | 0 | 11 250 € |
| Random Forest, 16 variables (validation imbriquée) | 0,560 ± 0,041 | 7 123 € |
| XGBoost, 28 variables (validation imbriquée) | 0,604 ± 0,033 | 6 816 € |

Le gain de R² est de +0,044, positif sur les 10 mesures. L'erreur est inférieure de 39 % à celle de la référence naïve.

Un modèle Ridge entraîné sur les mêmes variables sert à lire les effets en euros, à caractéristiques égales : environ +5 300 € par an pour une piscine, +4 600 € pour une villa, +4 100 € pour un lave-vaisselle, +3 100 € par chambre supplémentaire. Ce sont des associations statistiques, pas des effets causaux.

## Contenu du dépôt

| Chemin | Contenu |
|---|---|
| `notebooks/airbnb_market_intelligence.ipynb` | Le projet complet : nettoyage, exploration, modélisation, pistes testées, modèle amélioré |
| `app/app.py` | Application Streamlit (exploration du marché, modèles, simulateur) |
| `app/pipeline.py` | Nettoyage, construction des variables et définition des modèles, partagés par l'application |
| `app/preparer_modeles.py` | Évaluation et sauvegarde des modèles utilisés par l'application |
| `app/data/riviera_du_levant.csv` | Export brut AirROI |
| `app/artefacts/` | Modèles entraînés et métriques |

## Installation

Python 3.10 ou plus récent.

    pip install -r requirements.txt

## Lancer l'application

    cd app
    streamlit run app.py

Les modèles sont fournis dans `app/artefacts/`. S'ils sont incompatibles avec les versions installées de scikit-learn ou de xgboost, l'application les entraîne de nouveau au démarrage. Pour tout recalculer, y compris la validation imbriquée (quelques minutes) :

    python preparer_modeles.py

## Exécuter le notebook

Le notebook lit le fichier CSV dans le dossier indiqué par la variable d'environnement `PROJECT_DIR`. Depuis la racine du dépôt :

    PROJECT_DIR=app/data jupyter notebook notebooks/airbnb_market_intelligence.ipynb

Sur Google Colab, il monte le Drive et lit le dossier défini en tête de la première cellule ; il suffit d'adapter ce chemin.

## Limites

- Les données sont observationnelles : le projet mesure des associations.
- Seules les annonces déjà actives (au moins 110 nuits réservées) sont observées. Le modèle ne dit rien du risque de ne pas louer.
- Le prix par nuit, la saisonnalité, la localisation précise et les notes des voyageurs ne sont pas dans l'export.
- Les revenus les plus élevés restent sous-estimés, et les biens au-delà de 82 495 € sont hors périmètre.
- Les mots-clés viennent du titre rédigé par l'hôte : ils reflètent à la fois le bien et la façon dont il est présenté.

## Pistes

Ajouter les champs disponibles dans l'API mais absents de l'export (type de bien déclaré, notes, coordonnées), calculer une distance à la plage, étendre l'analyse de texte aux descriptions, intégrer les coûts pour passer du revenu au rendement, et produire des intervalles de prédiction.

## Auteur

Marvin Jernidier, MJ Metrix.
