"""Assistant de l'application : répond en français à des questions business et data sur le projet.

Principe : aucun texte n'est rédigé par un modèle de langage. Chaque réponse est un gabarit rempli par
un calcul pandas sur les annonces nettoyées, par le modèle XGBoost ou par les métriques enregistrées
(artefacts/metriques.json). Les chiffres sont donc exactement ceux du notebook et des autres pages.

La compréhension de la question se fait en deux niveaux :
  1. Jev (TypeSafe), si une clé API est disponible : il classe la question parmi les sujets connus
     et renvoie une probabilité calibrée. Une question hors sujet est refusée poliment.
  2. Des règles par mots-clés, sans réseau : elles servent de secours et extraient toujours
     les paramètres (commune, nombre de chambres, équipements, titre entre guillemets).
Les réponses de Jev sont mises en cache : une question déjà posée ne coûte plus rien.
"""
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P

# ------------------------------------------------------------------ mise en forme


def eur(v, signe=False):
    s = f"{abs(v):,.0f}".replace(",", " ")
    if signe:
        return f"{'+' if v >= 0 else '−'}{s} €"
    return f"{'−' if v < 0 else ''}{s} €"


def dec(v, n=3, signe=False):
    s = f"{v:+.{n}f}" if signe else f"{v:.{n}f}"
    return s.replace(".", ",").replace("-", "−")


def entier(v):
    return f"{int(v):,}".replace(",", " ")


def pl(n, mot, pluriel=None):
    """« 1 chambre », « 2 chambres », « 1,5 salle de bain »."""
    nombre = f"{n:g}".replace(".", ",")
    return f"{nombre} {mot if n < 2 else (pluriel or mot + 's')}"


def liste_fr(elements):
    elements = list(elements)
    return elements[0] if len(elements) == 1 else ", ".join(elements[:-1]) + " et " + elements[-1]


def a_commune(c):
    return "au Gosier" if c == "Le Gosier" else f"à {c}"


def pct(v, n=0):
    return f"{v * 100:.{n}f} %".replace(".", ",")


def en_nombre(mot):
    """« 3 » → 3, « 1,5 » → 1.5, « deux » → 2."""
    if mot[0].isdigit():
        return float(mot.replace(",", ".")) if ("," in mot or "." in mot) else int(mot)
    return NOMBRES[mot]


def sans_accents(texte):
    t = unicodedata.normalize("NFD", texte.lower().replace("’", "'"))
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


# ------------------------------------------------------------------ vocabulaire

NOMBRES = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7}
MOTIF_NB = r"(\d+|une?|deux|trois|quatre|cinq|six|sept)"

COMMUNES_MOTIFS = {
    "Le Gosier": r"gosier",
    "Sainte-Anne": r"(sainte|ste)[ -]?anne",
    "Saint-François": r"(saint|st)[ -]?francois",
}
AUTRES_LIEUX = r"\b(paris|lyon|marseille|nice|bordeaux|martinique|reunion|deshaies|bouillante|pointe[ -]a[ -]pitre|" \
               r"basse[ -]terre|saint[ -]martin|saint[ -]barth|marie[ -]galante|les saintes|baie[ -]mahault|" \
               r"le moule|petit[ -]bourg|new york|londres|espagne|italie|corse|bretagne)\b"

# Variables du modèle reconnues dans une question (l'ordre compte pour les motifs qui se chevauchent)
VARIABLES_MOTIFS = [
    ("pool", r"piscine|\bpool"),
    ("air_conditioning", r"\bclim|climatis|air conditionne"),
    ("dishwasher", r"lave[ -]?vaisselle|dishwasher"),
    ("dedicated_workspace", r"espace de travail|coin bureau|\bbureau\b|teletravail|workspace"),
    ("waterfront", r"front de mer|bord de mer|pieds? dans l'eau|waterfront|bord de plage"),
    ("hot_tub", r"jacuzzi|\bspa\b|hot tub|bain a remous"),
    ("bbq_grill", r"barbecue|\bbbq\b|plancha"),
    ("cable_tv", r"\btv\b|\btele\b|television|\bcable"),
    ("gym", r"salle de sport|\bgym|fitness|musculation"),
    ("superhost", r"superhost|super[ -]?hote"),
    ("num_reviews", r"nombre d'avis|des avis|les avis|d'avis|\breviews?\b|commentaires|evaluations"),
    ("baths", r"salles? de bains?|salles? d'eau|\bsdb\b"),
    ("guests", r"voyageurs|capacite|couchages|nombre de personnes"),
    ("t_sea", r"vue mer|vue sur (la )?mer|vue ocean|vue sur l'ocean|vue lagon"),
    ("t_luxe", r"\bluxe|haut de gamme|prestige|standing"),
    ("t_villa", r"\bvillas?\b"),
    ("t_beach", r"\bplage"),
    ("bedrooms", r"chambres?|bedrooms?"),
]
VARIABLES_ACTIONNABLES = P.EQUIPEMENTS + ["superhost"]

TYPES_MOTIFS = [("Villa", r"\bvilla"), ("Studio", r"\bstudio"), ("Maison", r"\bmaison|\bcase\b"),
                ("Bungalow, gîte", r"bungalow|\bgite|\bkaz\b|cottage"), ("Appartement", r"appart|\bflat\b|\b[tf][1-7]\b")]

LIBELLES_INTENTIONS = {
    "marche": "vue d'ensemble du marché", "communes": "revenus par commune", "chambres": "revenus selon la taille",
    "impact": "effet d'un équipement ou d'une caractéristique", "estimation": "estimation d'un logement",
    "conseils": "conseils au propriétaire", "titre": "titre de l'annonce", "top": "annonces les plus performantes",
    "performance": "précision du modèle", "comparaison": "comparaison des modèles", "importance": "variables les plus importantes",
    "limites": "limites et erreurs du modèle", "donnees": "données et nettoyage", "validation": "méthode de validation",
    "jev": "Jev et son usage", "description": "profil type des logements", "donnee_absente": "information absente des données", "aide": "aide",
    "hors_perimetre": "hors du périmètre du projet",
}

# Descriptions données à Jev pour classer la question (une seule question fermée, plusieurs réponses possibles)
CRITERES_JEV = {
    "marche": "vue d'ensemble du marché : nombre d'annonces, revenu médian ou moyen de toute la zone, dispersion des revenus",
    "communes": "comparer les revenus entre les communes Le Gosier, Sainte-Anne, Saint-François, ou connaître le revenu d'une commune",
    "chambres": "revenu selon le nombre de chambres ou la taille du logement (studio, T2, T3, grande villa)",
    "impact": "effet d'un équipement ou d'une caractéristique sur le revenu (piscine, climatisation, lave-vaisselle, jacuzzi, front de mer, superhost, avis, salles de bain), ou classement des équipements",
    "estimation": "estimer le revenu annuel d'un logement précis décrit dans la question, combien il pourrait rapporter",
    "conseils": "conseils à un propriétaire ou à un investisseur : quoi améliorer en priorité, comment gagner plus, où et quoi acheter",
    "titre": "le titre de l'annonce : mots-clés du titre, audit ou note d'un titre, analyse sémantique des titres",
    "top": "les annonces qui gagnent le plus, les meilleurs exemples, le revenu record",
    "performance": "précision du modèle : R², erreur absolue moyenne, fiabilité des estimations, comparaison à une estimation naïve",
    "comparaison": "comparaison des algorithmes testés (XGBoost, Random Forest, Ridge, HistGradientBoosting) et raison du choix final",
    "importance": "variables les plus importantes pour le modèle, ce qui pèse le plus dans la prédiction",
    "limites": "limites et biais de l'étude, erreurs du modèle selon le niveau de revenu, causalité, domaine de validité",
    "donnees": "source des données (AirROI), nettoyage, annonces retirées, variable cible, période couverte",
    "validation": "méthode d'évaluation : validation croisée imbriquée, surapprentissage, réglage des hyperparamètres",
    "jev": "le modèle Jev de TypeSafe : ce qu'il est, son coût et son usage dans ce projet",
    "description": "caractéristiques des logements du marché : nombre moyen de voyageurs, de chambres ou de salles de bain, part des annonces avec piscine ou un autre équipement, part de Superhosts",
    "donnee_absente": "prix par nuit, taux d'occupation, saisonnalité, notes en étoiles laissées par les clients, coûts, rentabilité nette ou fiscalité",
    "aide": "salutation, remerciement, ou demande de ce que l'assistant sait faire",
    "hors_perimetre": "question sans rapport avec ce projet de locations saisonnières de la Riviera du Levant en Guadeloupe (météo, actualité, code informatique, culture générale, autre région)",
}

# Règles hors ligne : (intention, motif, poids)
REGLES = [
    ("aide", r"^(bonjour|salut|hello|bonsoir|coucou|merci)\b|que sais[- ]tu|que peux[- ]tu|quelles questions|comment (tu marches|ca marche)|\baide\b", 2),
    ("donnee_absente", r"prix (par|a la) nuit|prix de la nuit|tarif|taux d'occupation|occupation|saison|haute saison|basse saison|"
                       r"\bnotes?\b des voyageurs|note moyenne|\bcouts?\b|charges|rentabilite nette|rendement|fiscalit|impots?|credit|prix d'achat|prix au m", 3),
    ("jev", r"\bjev\b|typesafe|\bllm\b|intelligence artificielle|\bia\b", 3),
    ("titre", r"\btitres?\b|mots?[- ]cles|nom de l'annonce|intitule|text mining", 3),
    ("top", r"meilleures? annonces?|plus performantes?|gagnent le plus|record|top \d+|\btop\b|plus rentables? annonces?", 3),
    ("validation", r"validation croisee|imbriquee|(a|a-t-il|ont) ete validee?s?|comment .*valide|surapprentissage|sur-apprentissage|overfitting|hyperparam|randomizedsearch|gridsearch|plis\b|k-?fold", 3),
    ("comparaison", r"comparai?son des modeles|quels? modeles?|modeles? (testes|compares|essayes)|pourquoi (xgboost|ce modele)|random forest|ridge|"
                    r"histgradient|regression lineaire|algorithmes?|neurones?|deep learning|lightgbm|catboost", 2),
    ("performance", r"\br2\b|\br²|r carre|\bmae\b|erreur (absolue )?moyenne|precision|precis|fiabl|performance du modele|qualite du modele|score", 2),
    ("importance", r"importance|variables? (les plus )?importantes?|variables? (qui )?compt|compte(nt)? le plus|pese(nt)? le plus|permutation|shap|qu'est-ce qui (compte|explique)|facteurs? (cles|principaux)", 2),
    ("limites", r"limites?|biais|sous-?estim|sur-?estim|causal|faiblesses?|se trompe|erreurs? du modele|quintile|domaine de validite", 2),
    ("donnees", r"donnees|dataset|airroi|nettoyage|nettoye|valeurs? aberrantes?|extremes|source|variable cible|ttm|combien d'annonces|annonces retirees", 2),
    ("conseils", r"conseils?|ameliorer|augmenter|optimiser|booster|maximiser|que (dois|devrais)[- ]je|quoi faire|priorit|investir|"
                 r"investissement|acheter|recommand|gagner plus|valoriser|que faire|comment (gagner|faire)", 3),
    ("estimation", r"combien (pourrait|peut|va|rapporterait|rapporte|gagnerait|gagne)|\bestim|prevoir|predi|j'ai une? |mon (logement|appart|bien)|ma (villa|maison)", 2),
    ("impact", r"impact|effet|\bapporte|vaut[- ]il|vaut le coup|interet|influence|change|joue|ajouter|installer|equipements?", 2),
    ("communes", r"communes?|\bville|ou (faut[- ]il|investir|acheter)|quelle zone|localisation", 1),
    ("chambres", r"chambres?|taille|studio|\b[tf][1-7]\b|pieces", 1),
    ("description", r"part des|pourcentage|proportion|combien d'annonces (ont|avec|disposent)|combien de logements (ont|avec)|en moyenne combien de|"
                    r"nombre moyen de|taille moyenne|logement (moyen|type)|profil type", 3),
    ("marche", r"marche|revenu (median|moyen)|en moyenne|combien gagne|revenus? (des|de la) (annonces|zone)|distribution|vue d'ensemble|panorama", 1),
]


class Reponse:
    """Une réponse de l'assistant : le texte, un tableau facultatif et la source des chiffres.
    comprise_par et confiance sont remplis ensuite : qui a compris la question (Jev ou les règles) et avec quelle probabilité."""
    def __init__(self, texte, intention, tableau=None, source=""):
        self.texte = texte
        self.intention = intention
        self.tableau = tableau
        self.source = source
        self.comprise_par = "règles"
        self.confiance = None


# ------------------------------------------------------------------ client Jev


class Jev:
    """Appels à l'API Jev (TypeSafe) avec cache sur disque et budget d'appels.
    La clé est lue dans la variable d'environnement TYPESAFE_API_KEY (ou passée par st.secrets)."""
    URL = "https://api.typesafe.ai/v1/systemone"
    PRIX_JETON_USD = 0.034 / 808464  # coût réel mesuré lors de l'audit des titres (novembre 2026)

    def __init__(self, cle=None, chemin_cache=None, budget=150, modele="jev-latest"):
        self.cle = cle or os.environ.get("TYPESAFE_API_KEY")
        self.chemin_cache = Path(chemin_cache) if chemin_cache else None
        self.budget, self.modele = budget, modele
        self.appels, self.jetons, self.derniere_erreur = 0, 0, None
        self.cache = {}
        if self.chemin_cache and self.chemin_cache.exists():
            try:
                self.cache = json.loads(self.chemin_cache.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.cache = {}

    @property
    def disponible(self):
        return bool(self.cle) and self.appels < self.budget

    @property
    def cout_usd(self):
        return self.jetons * self.PRIX_JETON_USD

    def _sauver(self):
        if self.chemin_cache:
            try:
                self.chemin_cache.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8")
            except OSError:
                pass  # hébergement en lecture seule : le cache reste en mémoire

    def demander(self, cle_cache, etat, questions):
        """Renvoie (réponses de Jev, origine) ; origine vaut "cache" ou "api". Renvoie (None, None) si l'API est indisponible :
        pas de clé, budget atteint, réseau coupé. L'assistant passe alors aux règles, sans erreur affichée."""
        if cle_cache in self.cache:
            return self.cache[cle_cache], "cache"
        if not self.disponible:
            return None, None
        corps = json.dumps({"model": self.modele, "state": etat, "questions": questions}).encode()
        for essai in range(3):   # jusqu'à 3 essais si le service est momentanément saturé
            req = urllib.request.Request(self.URL, data=corps, headers={
                "Authorization": "Bearer " + self.cle, "Content-Type": "application/json"})
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=15).read())
                self.appels += 1
                u = r.get("usage", {})
                self.jetons += u.get("input_tokens", 0) + u.get("output_tokens", 0)
                self.cache[cle_cache] = r["answers"]
                self._sauver()
                return r["answers"], "api"
            except urllib.error.HTTPError as e:
                self.derniere_erreur = f"HTTP {e.code}"
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(1.5 * (essai + 1))
                    continue
                return None, None
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
                self.derniere_erreur = type(e).__name__
                return None, None
        return None, None


# ------------------------------------------------------------------ assistant


class Assistant:
    SEUIL_JEV = 0.45          # confiance minimale pour suivre le classement de Jev
    SEUIL_HORS_SUJET = 0.5
    CONTEXTE = ("Assistant d'une application de data science sur les locations saisonnières Airbnb (logements entiers) "
                "de la Riviera du Levant en Guadeloupe : Le Gosier, Sainte-Anne, Saint-François. Données AirROI, "
                "revenu annuel sur 12 mois, 1 163 annonces, équipements, modèle XGBoost de prédiction du revenu.")

    def __init__(self, df, xgb, ridge, metriques, coefs, echelle, jev_resultats=None, jev_titres=None, jev=None):
        self.df, self.xgb, self.ridge, self.M = df, xgb, ridge, metriques
        self.coefs, self.echelle = coefs, echelle
        self.JR, self.JT, self.jev = jev_resultats, jev_titres, jev
        self.imb = metriques["imbrique"]
        self.y = df[P.CIBLE]

    # ---------------------------------------------------------- extraction des paramètres
    def extraire(self, question):
        """Repère dans la question les paramètres utiles aux calculs, avec des expressions régulières.
        Renvoie un dictionnaire : communes citées, nombre de chambres, équipements, titre entre guillemets..."""
        q = sans_accents(question)
        ent = {"q": q, "communes": [c for c, m in COMMUNES_MOTIFS.items() if re.search(m, q)]}
        # Un titre d'annonce écrit entre guillemets est mis de côté pour ne pas fausser la lecture du reste
        titre = re.search(r"[«\"“]\s*([^»\"”]{4,120}?)\s*[»\"”]", question)
        ent["titre"] = titre.group(1).strip() if titre else None
        q_sans_titre = sans_accents(question.replace(titre.group(0), " ")) if titre else q
        ent["autre_lieu"] = bool(re.search(AUTRES_LIEUX, q_sans_titre))

        # Nombre de chambres : « 3 chambres », « deux chambres », « T2 » (= 1 chambre) ou « studio » (= 0)
        ch = None
        m = re.search(MOTIF_NB + r"\s*(chambres?|ch\b|bedrooms?)", q_sans_titre)
        if m:
            ch = en_nombre(m.group(1))
        else:
            m = re.search(r"\b[tf]([1-7])\b", q_sans_titre) or re.search(r"(\d)\s*pieces", q_sans_titre)
            if m:
                ch = int(m.group(1)) - 1
                ent["notation_t"] = m.group(0)
            elif re.search(r"\bstudios?\b", q_sans_titre):
                ch = 0
        # « une chambre de plus » : on demande l'effet d'une chambre, pas un logement d'une chambre
        ent["marginal"] = bool(re.search(r"chambres? (de plus|supplementaires?|en plus)|(ajouter|rajouter) une chambre", q_sans_titre))
        if ent["marginal"]:
            ch = None
        ent["bedrooms"] = ch
        m = re.search(MOTIF_NB + r"\s*(voyageurs|personnes|pers\b|couchages|places|invites)", q_sans_titre)
        ent["guests"] = en_nombre(m.group(1)) if m else None
        m = re.search(r"(\d+(?:[.,]5)?|une?|deux|trois|quatre)\s*(salles? de bains?|salles? d'eau|sdb)", q_sans_titre)
        ent["baths"] = en_nombre(m.group(1)) if m else None
        m = re.search(r"(\d+)\s*(avis|reviews|commentaires)", q_sans_titre)
        ent["num_reviews"] = int(m.group(1)) if m else None

        # Équipements et caractéristiques cités ; « sans piscine » va dans les négations
        ent["variables"], ent["negations"] = [], []
        for var, motif in VARIABLES_MOTIFS:
            for mm in re.finditer(motif, q_sans_titre):
                avant = q_sans_titre[max(0, mm.start() - 14):mm.start()]
                if re.search(r"(sans|pas de|pas d'|aucune?|ni)\s*(un |une |de |d')?$", avant):
                    ent["negations"].append(var)
                elif var not in ent["variables"]:
                    ent["variables"].append(var)
        if re.search(r"(pas|non|sans|jamais)\s*(encore )?(superhost|super[ -]?hote)|nouvelle annonce|jamais (ete )?loue|pas encore (en )?location", q_sans_titre):
            ent["negations"].append("superhost")
            if "superhost" in ent["variables"]:
                ent["variables"].remove("superhost")
        ent["type_de_bien"] = next((t for t, m in TYPES_MOTIFS if re.search(m, q_sans_titre)), None)
        return ent

    # ---------------------------------------------------------- compréhension
    def regles(self, question, ent):
        """Compréhension sans réseau : chaque sujet a une liste de mots-clés et un poids (liste REGLES).
        On additionne les poids des motifs trouvés, on ajoute quelques bonus selon les paramètres repérés,
        et le sujet qui a le plus gros score l'emporte."""
        q = sans_accents(question)
        # « sous-estime » ne doit pas déclencher une estimation : on colle le mot avant de chercher « estim »
        q = q.replace("sous-estim", "sousestim").replace("sur-estim", "surestim")
        scores = {}
        for intention, motif, poids in REGLES:
            if re.search(motif, q):
                scores[intention] = scores.get(intention, 0) + poids
        actionnables = [v for v in ent["variables"] if v != "bedrooms"]
        profil = (ent["communes"] or ent["guests"] is not None or ent["type_de_bien"] or actionnables
                  or ent["negations"]) and (ent["bedrooms"] is not None or ent["guests"] is not None or ent["type_de_bien"] or actionnables)
        if ent["titre"]:
            scores["titre"] = scores.get("titre", 0) + 4
        if actionnables:
            scores["impact"] = scores.get("impact", 0) + 1
        if (profil and "conseils" not in scores and "limites" not in scores and re.search(r"combien|estim|rapport|gagner|revenu|esperer", q)
                and ("estimation" in scores or re.search(r"j'ai|mon |ma |un |une ", q))):
            scores["estimation"] = scores.get("estimation", 0) + 2
        if ent["communes"] and not profil and ent["bedrooms"] is None:
            scores["communes"] = scores.get("communes", 0) + 2
        if ent["communes"] and len(ent["communes"]) > 1:
            scores["communes"] = scores.get("communes", 0) + 2
        if ent["bedrooms"] is not None and "estimation" not in scores and "conseils" not in scores:
            scores["chambres"] = scores.get("chambres", 0) + 1
        if not scores:
            return None, 0.0
        meilleur = max(scores, key=lambda k: (scores[k], -list(LIBELLES_INTENTIONS).index(k)))
        if meilleur == "estimation" and not profil and ent["bedrooms"] is not None:
            meilleur = "chambres"   # « combien rapporte un 4 chambres ? » : les annonces réelles répondent mieux
        return meilleur, scores[meilleur]

    def comprendre(self, question, ent):
        """Renvoie (intention, comprise_par, confiance, autres intentions plausibles).
        Jev est consulté s'il est disponible ; sinon, ou en cas de panne, ce sont les règles qui décident."""
        locale, score = self.regles(question, ent)
        if ent["autre_lieu"] and not ent["communes"]:
            return "hors_perimetre", "règles", None, []
        if self.jev is not None:
            rep, origine = self.jev.demander(
                "intention|" + sans_accents(question).strip(),
                {"contexte": self.CONTEXTE, "question": question},
                {"sujet": {"type": "choice", "instructions": "De quoi parle cette question posée à l'assistant ?",
                           "criteria": CRITERES_JEV}})
            if rep:
                # Jev renvoie une probabilité par sujet : on les trie de la plus forte à la plus faible
                probas = rep["sujet"]["probabilities"]
                ordre = sorted(probas, key=probas.get, reverse=True)
                top, conf = ordre[0], probas[ordre[0]]
                par = "Jev" if origine == "api" else "Jev (cache)"
                if top == "hors_perimetre" and conf >= self.SEUIL_HORS_SUJET and score < 3:
                    return "hors_perimetre", par, conf, []
                if top == "hors_perimetre":
                    ordre = ordre[1:]
                    top, conf = ordre[0], probas[ordre[0]]
                if conf >= self.SEUIL_JEV:
                    return top, par, conf, []
                # Jev hésite : on garde l'avis des règles s'il fait partie de ses trois premiers choix, sinon on demande de reformuler
                if locale and locale in ordre[:3]:
                    return locale, par + " + règles", probas[locale], []
                return None, par, conf, [o for o in ordre[:3] if o not in ("aide", "hors_perimetre")]
        return locale, "règles", None, []

    # ---------------------------------------------------------- point d'entrée
    def repondre(self, question):
        question = (question or "").strip()[:400]
        if not question:
            return self.r_aide({})
        ent = self.extraire(question)
        intention, par, conf, pistes = self.comprendre(question, ent)
        if intention is None:
            if pistes:
                r = Reponse("Je ne suis pas sûr de comprendre. Votre question porte-t-elle plutôt sur : "
                            + ", ".join(LIBELLES_INTENTIONS[p] for p in pistes) + " ? Vous pouvez la reformuler "
                            "ou choisir une des questions proposées.", "incertain")
            else:
                r = self.r_hors_perimetre(ent, compris=False)
        else:
            if ent["marginal"] and intention in ("chambres", "estimation", "marche"):
                intention = "impact"
            if intention == "conseils" and ent["communes"] and ent["bedrooms"] is None and not ent["variables"] and not ent["type_de_bien"]:
                intention = "communes"   # « Gosier ou Saint-François pour investir ? »
            reponses = {"marche": self.r_marche, "communes": self.r_communes, "chambres": self.r_chambres,
                        "impact": self.r_impact, "estimation": self.r_estimation, "conseils": self.r_conseils,
                        "titre": self.r_titre, "top": self.r_top, "description": self.r_description,
                        "performance": self.r_performance, "comparaison": self.r_comparaison,
                        "importance": self.r_importance, "limites": self.r_limites, "donnees": self.r_donnees,
                        "validation": self.r_validation, "jev": self.r_jev, "donnee_absente": self.r_donnee_absente,
                        "aide": self.r_aide, "hors_perimetre": self.r_hors_perimetre}
            r = reponses[intention](ent)   # chaque sujet a sa fonction de réponse
            r.intention = intention
        r.comprise_par, r.confiance = par, conf
        return r

    # ---------------------------------------------------------- outils de calcul
    def _stats(self, s):
        q1, med, q3 = s.quantile([.25, .5, .75])
        return {"n": len(s), "mediane": med, "moyenne": s.mean(), "q1": q1, "q3": q3}

    def _profil(self, ent):
        ch = 1 if ent["bedrooms"] is None else ent["bedrooms"]
        meme = self.df[self.df.bedrooms == min(ch, 5)]
        guests = ent["guests"] if ent["guests"] is not None else int(meme.guests.median())
        baths = ent["baths"] if ent["baths"] is not None else round(float(meme.baths.median()) * 2) / 2
        type_de_bien = ent["type_de_bien"] or ("Studio" if ch == 0 else "Autre")
        profil = {"bedrooms": ch, "guests": guests, "baths": max(baths, 1.0), "type_de_bien": type_de_bien,
                  "superhost": 0 if "superhost" in ent["negations"] else 1,
                  "num_reviews": ent["num_reviews"] if ent["num_reviews"] is not None else 65,
                  "air_conditioning": 0 if "air_conditioning" in ent["negations"] else 1}
        for v in ent["variables"]:
            if v in P.EQUIPEMENTS:
                profil[v] = 1
        hypotheses = []
        if ent["bedrooms"] is None:
            hypotheses.append("1 chambre (non précisé)")
        if ent.get("notation_t"):
            hypotheses.append(f"{ent['notation_t'].upper()} lu comme {pl(ch, 'chambre')}")
        if ent["guests"] is None:
            hypotheses.append(f"{guests} voyageurs (valeur médiane pour cette taille)")
        if ent["baths"] is None:
            hypotheses.append(f"{pl(max(baths, 1.0), 'salle de bain', 'salles de bain')} (médiane)")
        hypotheses.append("Superhost" if profil["superhost"] else "pas Superhost")
        if ent["num_reviews"] is None:
            hypotheses.append("65 avis (valeur du simulateur)")
        if "air_conditioning" not in ent["variables"] and profil["air_conditioning"]:
            hypotheses.append("climatisation (présente dans 99 % des annonces)")
        return profil, hypotheses

    def _predire(self, profil):
        return float(self.xgb.predict(P.profil_vers_X(profil))[0])

    def _equipements_cites(self, profil):
        return [P.LIBELLES[e].lower() for e in P.EQUIPEMENTS if profil.get(e) and e != "air_conditioning"]

    # ---------------------------------------------------------- réponses business
    def r_description(self, ent):
        d = self.df if not ent["communes"] else self.df[self.df.locality.isin(ent["communes"])]
        lieu = " " + " et ".join(a_commune(c) for c in ent["communes"]) if ent["communes"] else ""
        lignes = [("Chambres", f"{dec(d.bedrooms.mean(), 2)} en moyenne, médiane {d.bedrooms.median():g}"),
                  ("Voyageurs", f"{dec(d.guests.mean(), 1)} en moyenne, médiane {d.guests.median():g}"),
                  ("Salles de bain", f"{dec(d.baths.mean(), 2)} en moyenne, médiane {d.baths.median():g}".replace(".", ",")),
                  ("Nombre d'avis", f"{dec(d.num_reviews.mean(), 0)} en moyenne, médiane {d.num_reviews.median():g}"),
                  ("Superhost", pct(d.superhost.mean()))]
        lignes += [(P.LIBELLES[e], pct(d[e].mean())) for e in P.EQUIPEMENTS]
        t = pd.DataFrame(lignes, columns=["Caractéristique", "Annonces" + lieu])
        demandees = [v for v in ent["variables"] if v in P.EQUIPEMENTS + ["superhost"]]
        if demandees:
            texte = " ".join(f"{P.LIBELLES[v]} : {entier(int(d[v].sum()))} annonces sur {entier(len(d))}{lieu}, soit {pct(d[v].mean())}." for v in demandees)
        else:
            texte = (f"Le logement type{lieu} compte {dec(d.bedrooms.mean(), 2)} chambres et accueille {dec(d.guests.mean(), 1)} voyageurs en moyenne "
                     f"(médianes : {d.bedrooms.median():g} chambre et {d.guests.median():g} voyageurs). {pct(d.pool.mean())} des annonces ont une piscine, "
                     f"{pct(d.air_conditioning.mean())} la climatisation et {pct(d.superhost.mean())} sont tenues par un Superhost.")
        return Reponse(texte, "description", t, "Calcul pandas sur les annonces nettoyées.")

    def r_marche(self, ent):
        s = self._stats(self.y)
        p90 = self.y.quantile(.9)
        texte = (f"Le marché étudié compte {entier(s['n'])} logements entiers actifs sur la Riviera du Levant. "
                 + f"Leur revenu annuel médian est de {eur(s['mediane'])} et la moyenne de {eur(s['moyenne'])} : "
                 f"quelques gros revenus tirent la moyenne vers le haut. La moitié des annonces gagne entre {eur(s['q1'])} "
                 f"et {eur(s['q3'])} par an, et les 10 % les plus performantes dépassent {eur(p90)}.")
        tab = self._tableau_communes()
        return Reponse(texte, "marche", tab, "Calcul pandas sur les annonces nettoyées (revenu des 12 derniers mois, AirROI).")

    def _tableau_communes(self):
        g = self.df.groupby("locality")
        t = pd.DataFrame({"Annonces": g.size(), "Revenu médian": g[P.CIBLE].median(), "Revenu moyen": g[P.CIBLE].mean(),
                          "Chambres en moyenne": g.bedrooms.mean(), "Avec piscine": g.pool.mean()}).reindex(P.COMMUNES)
        t = t.sort_values("Revenu médian", ascending=False)
        out = pd.DataFrame({"Commune": t.index, "Annonces": t["Annonces"].values,
                            "Revenu médian": t["Revenu médian"].map(eur).values,
                            "Revenu moyen": t["Revenu moyen"].map(eur).values,
                            "Chambres en moyenne": t["Chambres en moyenne"].map(lambda v: dec(v, 2)).values,
                            "Avec piscine": t["Avec piscine"].map(pct).values})
        return out

    def _ajuste_commune(self, c):
        return 0.0 if c == "Le Gosier" else self.coefs[f"locality_{c}"]

    def r_communes(self, ent):
        g = self.df.groupby("locality")[P.CIBLE].median().reindex(P.COMMUNES)
        ch = self.df.groupby("locality").bedrooms.mean()
        ordre = g.sort_values(ascending=False)
        source = "Calcul pandas sur les annonces nettoyées ; écart à caractéristiques égales : coefficients du modèle Ridge (28 variables)."
        cibles = ent["communes"] if ent["communes"] else []
        if len(cibles) == 1:
            c = cibles[0]
            s = self._stats(self.df[self.df.locality == c][P.CIBLE])
            rang = list(ordre.index).index(c) + 1
            rangs = {1: "le plus élevé des trois communes", 2: "au deuxième rang des trois communes", 3: "le plus bas des trois communes"}
            aj = self._ajuste_commune(c)
            texte = (f"{c} compte {s['n']} annonces. Revenu annuel médian : {eur(s['mediane'])} ({rangs[rang]}), "
                     f"moyenne {eur(s['moyenne'])}, la moitié des annonces entre {eur(s['q1'])} et {eur(s['q3'])}. "
                     f"Les logements y ont {dec(ch[c], 2)} chambres en moyenne. ")
            if c == "Le Gosier":
                texte += ("Le Gosier sert de référence au modèle : à caractéristiques égales, Saint-François est estimé à "
                          f"{eur(self.coefs['locality_Saint-François'], True)} et Sainte-Anne à {eur(self.coefs['locality_Sainte-Anne'], True)} par rapport au Gosier.")
            else:
                texte += f"À caractéristiques égales, le modèle Ridge estime {c} à {eur(aj, True)} par an par rapport au Gosier."
            return Reponse(texte, "communes", self._tableau_communes(), source)
        if len(cibles) == 2:
            c1, c2 = sorted(cibles, key=lambda c: -g[c])
            aj = self._ajuste_commune(c1) - self._ajuste_commune(c2)
            texte = (f"En revenu médian, {c1} dépasse {c2} de {eur(g[c1] - g[c2])} ({eur(g[c1])} contre {eur(g[c2])}). "
                     f"Les logements y ont {dec(ch[c1], 2)} chambres en moyenne contre {dec(ch[c2], 2)}. "
                     f"À caractéristiques égales, le modèle Ridge estime {c1} à {eur(aj, True)} par an par rapport {a_commune(c2)}. ")
            if np.sign(aj) < 0:
                texte += "L'écart brut s'explique donc par la taille et les équipements des logements, pas par la commune elle-même."
            return Reponse(texte, "communes", self._tableau_communes(), source)
        noms = list(ordre.index)
        texte = (f"En revenu médian, {noms[0]} arrive en tête ({eur(ordre.iloc[0])}), devant {noms[1]} ({eur(ordre.iloc[1])}) "
                 f"et {noms[2]} ({eur(ordre.iloc[2])}). ")
        ecart_brut_sf = g["Saint-François"] - g["Le Gosier"]
        aj_sf = self.coefs["locality_Saint-François"]
        if np.sign(ecart_brut_sf) != np.sign(aj_sf):
            texte += (f"Mais cet écart vient surtout des logements eux-mêmes : à Saint-François ils ont {dec(ch['Saint-François'], 2)} chambres "
                      f"en moyenne contre {dec(ch['Le Gosier'], 2)} au Gosier. À caractéristiques égales, le modèle Ridge inverse le classement : "
                      f"Saint-François {eur(aj_sf, True)} et Sainte-Anne {eur(self.coefs['locality_Sainte-Anne'], True)} par rapport au Gosier. "
                      f"La commune pèse peu dans la prédiction : ce sont la taille et les équipements qui font le revenu.")
        else:
            texte += (f"À caractéristiques égales, le modèle Ridge estime Saint-François à {eur(aj_sf, True)} et Sainte-Anne à "
                      f"{eur(self.coefs['locality_Sainte-Anne'], True)} par rapport au Gosier.")
        return Reponse(texte, "communes", self._tableau_communes(), source)

    def r_chambres(self, ent):
        d = self.df if not ent["communes"] else self.df[self.df.locality.isin(ent["communes"])]
        lieu = " " + " et ".join(a_commune(c) for c in ent["communes"]) if ent["communes"] else ""
        source = "Calcul pandas sur les annonces nettoyées. Les annonces sans nombre de chambres sont des studios (0 chambre)."
        tranche = d.bedrooms.clip(upper=5)
        g = d.groupby(tranche)[P.CIBLE]
        tab = pd.DataFrame({"Chambres": [("studio" if k == 0 else ("5 et plus" if k == 5 else str(int(k)))) for k in g.size().index],
                            "Annonces": g.size().values, "Revenu médian": g.median().map(eur).values,
                            "Moitié centrale": [f"{eur(a)} à {eur(b)}" for a, b in zip(g.quantile(.25), g.quantile(.75))]})
        n = ent["bedrooms"]
        if n is None:
            med = g.median()
            texte = (f"Le revenu suit nettement la taille{lieu} : médiane de {eur(med.get(0, np.nan))} pour un studio, "
                     f"{eur(med.get(1, np.nan))} pour 1 chambre, {eur(med.get(2, np.nan))} pour 2 et {eur(med.get(3, np.nan))} pour 3 chambres. "
                     f"À autres caractéristiques égales, le modèle Ridge associe chaque chambre supplémentaire à "
                     f"{eur(self.coefs['bedrooms'] / self.echelle['bedrooms']['iqr'], True)} par an. "
                     "C'est, avec les salles de bain, la variable dont le modèle XGBoost a le plus besoin.")
            return Reponse(texte, "chambres", tab, source)
        sel = d[d.bedrooms == n] if n < 5 else d[d.bedrooms >= 5]
        nom = "studios" if n == 0 else (f"logements de {pl(n, 'chambre')}" if n < 5 else "logements de 5 chambres et plus")
        if ent.get("notation_t"):
            nom += f" ({ent['notation_t'].upper()})"
        if len(sel) == 0:
            return Reponse(f"Aucune annonce de {nom}{lieu} dans les données : je ne peux pas donner de chiffre fiable.", "chambres", tab, source)
        s = self._stats(sel[P.CIBLE])
        texte = (f"{s['n']} {nom}{lieu} dans les données. Revenu annuel médian : {eur(s['mediane'])}, "
                 f"moyenne {eur(s['moyenne'])}, la moitié d'entre eux entre {eur(s['q1'])} et {eur(s['q3'])}.")
        if s["n"] < 15:
            texte += " Attention, l'échantillon est petit : ces chiffres sont fragiles."
        if re.search(r"rentab|rendement", ent.get("q", "")):
            texte += " Il s'agit du revenu brut : sans prix d'achat ni coûts dans les données, la rentabilité nette ne peut pas être calculée."
        return Reponse(texte, "chambres", tab, source)

    def _phrase_variable(self, v, communes=None):
        lib = P.LIBELLES[v]
        c = self.coefs[v]
        if v in P.CONTINUES:
            iqr = self.echelle[v]["iqr"]
            unite = {"bedrooms": "chambre", "baths": "salle de bain", "guests": "voyageur", "num_reviews": "avis"}[v]
            par_unite = c / iqr
            texte = f"{lib} : à autres caractéristiques égales, le modèle Ridge associe chaque {unite} supplémentaire à {eur(par_unite, True)} par an"
            if v == "baths":
                texte += f" (soit {eur(c, True)} par demi-salle de bain, l'unité du modèle)"
            if v == "num_reviews":
                texte = (f"Nombre d'avis : à autres caractéristiques égales, {iqr:g} avis de plus sont associés à {eur(c, True)} par an "
                         "selon le modèle Ridge. Le nombre d'avis reflète surtout l'ancienneté et le volume de réservations, ce n'est pas un levier direct")
            rang = self._rang_permutation(v)
            return texte + f". C'est la variable n° {rang} sur 28 pour la précision du modèle XGBoost."
        d = self.df[self.df.locality.isin(communes)] if communes else self.df
        a, s = d[d[v] == 1], d[d[v] == 0]
        if len(a) < 10 or len(s) < 10:
            d, communes = self.df, None
            a, s = d[d[v] == 1], d[d[v] == 0]
        brut = a[P.CIBLE].mean() - s[P.CIBLE].mean()
        part = len(a) / len(d)
        marche = "du marché" if not communes else "des annonces " + " et ".join(a_commune(c).replace("à ", "de ").replace("au ", "du ") for c in communes)
        if v.startswith("t_"):
            debut = f"Mention « {lib.replace('Titre : ', '')} » dans le titre ({entier(len(a))} annonces, {pct(part)})"
        else:
            debut = (f"Superhost ({entier(len(a))} annonces, {pct(part)} {marche})" if v == "superhost"
                     else f"{lib} ({entier(len(a))} annonces équipées, {pct(part)} {marche})")
        texte = (f"{debut} : revenu moyen de {eur(a[P.CIBLE].mean())} contre {eur(s[P.CIBLE].mean())} sans, soit un écart brut de {eur(brut, True)}. ")
        dch = a.bedrooms.mean() - s.bedrooms.mean()
        if abs(dch) >= 0.15:
            texte += f"Mais ces logements ont aussi {dec(abs(dch), 1)} chambre {'de plus' if dch > 0 else 'de moins'} en moyenne. "
        texte += f"À caractéristiques égales{' (toutes communes confondues)' if communes else ''}, le modèle Ridge estime l'écart à {eur(c, True)} par an"
        rang = self._rang_permutation(v)
        texte += f" ; c'est la variable n° {rang} sur 28 pour la précision du modèle XGBoost."
        if part > 0.95:
            texte += f" Presque tous les logements l'ont ({pct(part)}) : l'absence est pénalisante, la présence ne distingue pas."
        return texte

    def _rang_permutation(self, v):
        pi = pd.Series(self.M["permutation"]).sort_values(ascending=False)
        return list(pi.index).index(v) + 1

    def _tableau_equipements(self):
        lignes = []
        for e in VARIABLES_ACTIONNABLES:
            a, s = self.df[self.df[e] == 1][P.CIBLE], self.df[self.df[e] == 0][P.CIBLE]
            lignes.append({"Équipement": P.LIBELLES[e], "Annonces équipées": pct(len(a) / len(self.df)),
                           "Écart brut (moyennes)": a.mean() - s.mean(), "À caractéristiques égales (Ridge)": self.coefs[e],
                           "Rang XGBoost": self._rang_permutation(e)})
        t = pd.DataFrame(lignes).sort_values("À caractéristiques égales (Ridge)", ascending=False)
        for c in ["Écart brut (moyennes)", "À caractéristiques égales (Ridge)"]:
            t[c] = t[c].map(lambda v: eur(v, True))
        return t

    def r_impact(self, ent):
        source = ("Écart brut : calcul pandas (moyenne avec − moyenne sans). À caractéristiques égales : coefficients du modèle Ridge. "
                  "Rang : permutation importance du modèle XGBoost sur les plis de validation.")
        variables = [v for v in ent["variables"] if v != "bedrooms"] or (["bedrooms"] if "bedrooms" in ent["variables"] else [])
        variables += [v for v in ent["negations"] if v in VARIABLES_ACTIONNABLES and v not in variables]
        avert = " Ce sont des associations observées sur des annonces existantes, pas des effets garantis, et le coût de l'équipement n'est pas pris en compte."
        if not variables:
            t = self._tableau_equipements()
            texte = (f"À caractéristiques égales, les équipements les plus associés au revenu sont la piscine "
                     f"({eur(self.coefs['pool'], True)} par an), le lave-vaisselle ({eur(self.coefs['dishwasher'], True)}) "
                     f"et la climatisation ({eur(self.coefs['air_conditioning'], True)}, mais 99 % des logements l'ont déjà). "
                     f"Le statut Superhost pèse {eur(self.coefs['superhost'], True)}. Les écarts bruts sont bien plus forts, "
                     "car les logements équipés sont aussi plus grands." + avert)
            return Reponse(texte, "impact", t, source)
        if not any(v in P.EQUIPEMENTS for v in variables[:3]):
            avert = " Ce sont des associations observées sur des annonces existantes, pas des effets garantis."
        texte = " ".join(self._phrase_variable(v, ent["communes"]) for v in variables[:3]) + avert
        return Reponse(texte, "impact", None, source)

    def r_estimation(self, ent):
        if ent["bedrooms"] is None and not ent["communes"] and not ent["variables"] and not ent["type_de_bien"] and ent["guests"] is None:
            return Reponse("Pour estimer un revenu, décrivez le logement : commune, nombre de chambres et équipements. Par exemple : "
                           "combien rapporte une villa de 3 chambres avec piscine à Sainte-Anne ? Pour un calcul pas à pas, "
                           "le Property Simulator (page 4) permet de régler chaque caractéristique.", "estimation")
        profil, hyp = self._profil(ent)
        source = "Prédiction du modèle XGBoost (28 variables) ; erreur typique : MAE en validation croisée imbriquée."
        mae = self.imb["xgb_mae"]
        eqs = self._equipements_cites(profil)
        desc = "un studio" if profil["bedrooms"] == 0 else f"un logement de {pl(profil['bedrooms'], 'chambre')}"
        if profil["type_de_bien"] not in ("Autre", "Studio"):
            article = "une" if profil["type_de_bien"] in ("Villa", "Maison") else "un"
            nom = "bungalow ou gîte" if profil["type_de_bien"].startswith("Bungalow") else profil["type_de_bien"].lower()
            desc = f"{article} {nom} de {pl(profil['bedrooms'], 'chambre')}"
        if eqs:
            desc += " avec " + liste_fr(eqs)
        alerte = ""
        if profil["bedrooms"] > 6:
            alerte = " Attention : plus de 6 chambres, c'est en dehors des logements observés, l'estimation n'est pas fiable."
        communes = ent["communes"] or P.COMMUNES
        lignes = []
        for c in communes:
            est = self._predire({**profil, "commune": c})
            comp = self.df[(self.df.locality == c) & (self.df.bedrooms == profil["bedrooms"])][P.CIBLE]
            lignes.append({"Commune": c, "Estimation XGBoost": est, "Comparables": len(comp),
                           "Médiane des comparables": comp.median() if len(comp) >= 15 else np.nan})
        t = pd.DataFrame(lignes)
        if len(communes) == 1:
            r = lignes[0]
            texte = f"Pour {desc} {a_commune(r['Commune'])}, le modèle estime un revenu annuel d'environ {eur(r['Estimation XGBoost'])}, à ± {eur(mae)} près en moyenne. "
            if r["Comparables"] >= 15:
                texte += f"Pour comparaison, les {r['Comparables']} annonces réelles de même taille dans la commune ont une médiane de {eur(r['Médiane des comparables'])}. "
            else:
                texte += (f"Seulement {pl(r['Comparables'], 'annonce comparable', 'annonces comparables')} dans la commune : "
                          "estimation à lire avec prudence. ")
        else:
            texte = (f"Pour {desc}, le modèle estime entre {eur(t['Estimation XGBoost'].min())} et {eur(t['Estimation XGBoost'].max())} "
                     f"par an selon la commune, à ± {eur(mae)} près en moyenne. ")
        texte += "Hypothèses retenues : " + ", ".join(hyp) + "." + alerte
        t["Estimation XGBoost"] = t["Estimation XGBoost"].map(eur)
        t["Médiane des comparables"] = t["Médiane des comparables"].map(lambda v: "trop peu d'annonces" if pd.isna(v) else eur(v))
        return Reponse(texte, "estimation", t, source)

    def r_conseils(self, ent):
        source = "Variation de la prédiction XGBoost en ajoutant chaque équipement au logement décrit ; coefficients Ridge pour comparaison."
        a_profil = ent["bedrooms"] is not None or ent["communes"] or ent["variables"] or ent["negations"]
        jt = ""
        if self.JT is not None:
            n_pool = int(self.JT["pool"].sum())
            oubli = int(((self.JT["pool"] == 1) & (self.JT["t_pool"] == 0)).sum())
            jt = (f" Côté annonce, un point gratuit : {oubli} logements sur {n_pool} avec piscine ne la citent pas dans leur titre, "
                  f"et une vue mer mentionnée dans le titre est associée à {eur(self.coefs['t_sea'], True)} par an à caractéristiques égales.")
        avert = " Ces écarts sont des associations statistiques : à mettre en face du coût de chaque investissement."
        if not a_profil:
            texte = (f"Trois leviers ressortent à caractéristiques égales (modèle Ridge) : la piscine ({eur(self.coefs['pool'], True)} par an), "
                     f"le lave-vaisselle ({eur(self.coefs['dishwasher'], True)}), peu coûteux, et le statut Superhost ({eur(self.coefs['superhost'], True)}), "
                     f"qui dépend de la qualité d'accueil. La taille reste le premier facteur : chaque chambre est associée à "
                     f"{eur(self.coefs['bedrooms'], True)} et chaque demi-salle de bain à {eur(self.coefs['baths'], True)}."
                     + jt + avert + " Décrivez votre logement (commune, chambres, équipements) pour un conseil chiffré sur mesure.")
            return Reponse(texte, "conseils", self._tableau_equipements(), "Coefficients du modèle Ridge et audit des titres par Jev.")
        profil, hyp = self._profil(ent)
        communes = ent["communes"][:1] or ["Le Gosier"]
        profil["commune"] = communes[0]
        base = self._predire(profil)
        lignes = []
        for e in VARIABLES_ACTIONNABLES:
            if e != "waterfront" and not profil.get(e):   # le front de mer ne s'ajoute pas
                lignes.append({"Amélioration": ("Devenir Superhost" if e == "superhost" else "Ajouter : " + P.LIBELLES[e].lower()),
                               "XGBoost": self._predire({**profil, e: 1}) - base, "Ridge": self.coefs[e]})
        t = pd.DataFrame(lignes).sort_values("XGBoost", ascending=False)
        top = t.head(3)
        texte = (f"Pour ce logement ({'studio' if profil['bedrooms'] == 0 else pl(profil['bedrooms'], 'chambre')} {a_commune(communes[0])}), "
                 f"le modèle estime aujourd'hui {eur(base)} par an. Les améliorations les plus associées à un revenu supérieur : "
                 + liste_fr([f"{r['Amélioration'].replace('Ajouter : ', '').lower()} ({eur(r['XGBoost'], True)})" for _, r in top.iterrows()]) + ".")
        if not ent["communes"]:
            texte += " Commune non précisée : calcul fait pour Le Gosier."
        texte += " Hypothèses : " + ", ".join(hyp) + "." + jt + avert
        for c in ["XGBoost", "Ridge"]:
            t[c] = t[c].map(lambda v: eur(v, True))
        return Reponse(texte, "conseils", t.rename(columns={"XGBoost": "Gain estimé (XGBoost)", "Ridge": "Moyenne à caractéristiques égales (Ridge)"}), source)

    def r_top(self, ent):
        d = self.df if not ent["communes"] else self.df[self.df.locality.isin(ent["communes"])]
        if ent["bedrooms"] is not None:
            d = d[d.bedrooms == ent["bedrooms"]]
        if d.empty:
            return Reponse("Aucune annonce ne correspond à ces critères.", "top")
        t = d.sort_values(P.CIBLE, ascending=False).head(5)
        tab = pd.DataFrame({"Annonce": t.listing_name.values, "Commune": t.locality.values,
                            "Revenu annuel": t[P.CIBLE].map(eur).values, "Chambres": t.bedrooms.astype(int).values,
                            "Piscine": t.pool.map({1: "oui", 0: "non"}).values, "Superhost": t.superhost.map({1: "oui", 0: "non"}).values})
        p = t.iloc[0]
        texte = (f"L'annonce la plus performante{' de la sélection' if len(d) < len(self.df) else ''} est « {p.listing_name} » "
                 f"{a_commune(p.locality)} : {eur(p[P.CIBLE])} sur 12 mois, {pl(int(p.bedrooms), 'chambre')}. Sur les cinq premières, "
                 f"{int(t.pool.sum())} ont une piscine et {int(t.superhost.sum())} sont Superhost, pour {dec(t.bedrooms.mean(), 1)} chambres en moyenne.")
        return Reponse(texte, "top", tab, "Calcul pandas sur les annonces nettoyées (les revenus au-delà de 82 495 € ont été écartés au nettoyage).")

    def r_titre(self, ent):
        if ent["titre"]:
            return self._audit_titre(ent["titre"])
        texte = (f"Douze mots-clés sont extraits du titre de chaque annonce. À caractéristiques égales, le mot « villa » est associé à "
                 f"{eur(self.coefs['t_villa'], True)} par an, une vue mer citée à {eur(self.coefs['t_sea'], True)} et un vocabulaire haut de gamme à "
                 f"{eur(self.coefs['t_luxe'], True)}. Ces mots-clés font passer le R² de {dec(self.imb['ref_r2'])} à {dec(self.imb['xgb_r2'])} "
                 "(validation croisée imbriquée, avec XGBoost et le recodage des studios).")
        if self.JR:
            v = {x["nom"]: x for x in self.JR["variantes"]}
            off, rep, plus = (v["Officiel : XGBoost + mots-clés du titre"], v["Jev à la place des mots-clés"], v["Officiel + variables Jev"])
            texte += (f" Les {entier(self.JR['titres'])} titres ont aussi été lus par Jev (5 questions chacun, {self.JR['duree_s']} secondes, "
                      f"{dec(self.JR['cout_usd'], 2)} $). Jev à la place des mots-clés donne le même R² ({dec(rep['r2'])} contre {dec(off['r2'])}) ; "
                      f"ajouté aux mots-clés, le gain ({dec(plus['r2'] - off['r2'], 3, True)}) n'est pas significatif.")
        texte += " Pour auditer un titre, écrivez-le entre guillemets, par exemple : analyse le titre « Villa vue mer avec piscine »."
        return Reponse(texte, "titre", None, "Coefficients du modèle Ridge, métriques de validation croisée et audit Jev (artefacts).")

    def _audit_titre(self, titre):
        bas = titre.lower()
        trouves = [P.LIBELLES[k].replace("Titre : ", "") for k, m in P.MOTS_CLES.items() if re.search(m, bas)]
        effet = sum(self.coefs[k] for k, m in P.MOTS_CLES.items() if re.search(m, bas))
        texte = (f"Titre « {titre} ». Mots-clés reconnus : {', '.join(trouves) if trouves else 'aucun'}. "
                 f"À caractéristiques égales, leur effet cumulé dans le modèle Ridge est de {eur(effet, True)} par an. ")
        source = "Mots-clés du modèle et coefficients Ridge."
        tab = None
        rep = None
        if self.jev is not None:
            ctx = ("Titre d'une annonce de location saisonnière (Airbnb) en Guadeloupe, sur la Riviera du Levant "
                   "(Le Gosier, Sainte-Anne, Saint-François). Le titre est souvent en anglais ou en français.")
            rep, origine = self.jev.demander("titre|" + titre.strip().lower(), {"contexte": ctx, "titre": titre}, QUESTIONS_TITRE)
        if rep:
            j = {"vue mer": rep["vue_mer"]["noul"], "plage": rep["plage"]["noul"], "luxe": rep["standing"]["probabilities"]["luxe"],
                 "attrait": rep["attrait"]["score"]}
            texte += (f"Lecture de Jev : vue mer {dec(j['vue mer'], 2)}, plage {dec(j['plage'], 2)}, gamme luxe {dec(j['luxe'], 2)}, "
                      f"attrait {dec(j['attrait'], 1)} sur 4 (clientèle visée : {CIBLES[rep['cible']['choice']]}). ")
            if self.JT is not None:
                part = (self.JT["j_attrait"] < j["attrait"]).mean()
                texte += f"Cet attrait dépasse celui de {pct(part)} des titres du marché. "
            conseils = []
            if j["attrait"] < 2:
                conseils.append("citer des arguments concrets (piscine privée, vue mer, plage à pied, commune)")
            if j["vue mer"] < 0.5 and j["plage"] < 0.5:
                conseils.append("mentionner la mer ou la plage si le logement en est proche")
            if conseils:
                texte += "Pistes : " + " ; ".join(conseils) + "."
            tab = pd.DataFrame({"Question posée à Jev": ["Vue mer", "Plage", "Gamme luxe", "Attrait (0 à 4)"],
                                "Réponse": [dec(j["vue mer"], 2), dec(j["plage"], 2), dec(j["luxe"], 2), dec(j["attrait"], 1)]})
            source = "Mots-clés du modèle, coefficients Ridge et lecture sémantique par Jev (probabilités calibrées)."
        else:
            texte += "La lecture sémantique par Jev demande une clé API ; elle n'est pas active ici."
        return Reponse(texte, "titre", tab, source)

    # ---------------------------------------------------------- réponses data
    def r_performance(self, ent):
        i = self.imb
        gain_naif = 1 - i["xgb_mae"] / self.M["baseline_mae"]
        texte = (f"Le modèle retenu, XGBoost avec les mots-clés du titre, obtient un R² de {dec(i['xgb_r2'])} (± {dec(i['xgb_r2_std'])}) "
                 f"et une erreur absolue moyenne de {eur(i['xgb_mae'])}, mesurés en validation croisée imbriquée. Il explique donc environ "
                 f"{pct(i['xgb_r2'])} de la variance du revenu. Une estimation naïve (prédire la moyenne) se tromperait de {eur(self.M['baseline_mae'])} : "
                 f"le modèle réduit l'erreur de {pct(gain_naif)}. Par rapport à la première version (R² {dec(i['ref_r2'])}, {eur(i['ref_mae'])}), "
                 f"le gain de R² est de {dec(i['gain_r2'], 3, True)}, positif sur {i['plis_gagnes']} mesures sur {i['plis']}. "
                 f"Pour un bien jamais loué, sans Superhost ni avis, le modèle garde un R² de {dec(self.M['bien_neuf']['r2'])} "
                 f"et une erreur moyenne de {eur(self.M['bien_neuf']['mae'])}.")
        tab = pd.DataFrame({"Mesure": ["R² (validation croisée imbriquée)", "Erreur absolue moyenne", "Estimation naïve (MAE)", "Première version (R²)"],
                            "Valeur": [f"{dec(i['xgb_r2'])} ± {dec(i['xgb_r2_std'])}", eur(i["xgb_mae"]), eur(self.M["baseline_mae"]), dec(i["ref_r2"])]})
        return Reponse(texte, "performance", tab, "artefacts/metriques.json (validation croisée imbriquée, 5 plis × 2 répétitions).")

    def r_comparaison(self, ent):
        res = pd.DataFrame(self.M["resultats_cv"])
        t = pd.DataFrame({"Modèle": res.modele, "Variables": res.variables,
                          "R² (CV)": [f"{a} ± {b}" for a, b in zip(res.r2.map(dec), res.r2_std.map(dec))],
                          "MAE (CV)": res.mae.map(eur), "R² entraînement": res.r2_train.map(dec)})
        b = res.iloc[0]
        texte = (f"Six modèles ont été comparés en validation croisée à 5 plis. Le meilleur est {b.modele} : R² {dec(b.r2)} et erreur moyenne {eur(b.mae)}, "
                 f"devant Random Forest ({dec(res.iloc[1].r2)}) et Ridge ({dec(res.iloc[2].r2)}). XGBoost est retenu pour l'estimation, "
                 "et Ridge est gardé pour l'explication, car ses coefficients se lisent en euros. Ces scores sont un peu optimistes, car "
                 f"les mêmes plis ont servi au réglage : la mesure officielle, en validation croisée imbriquée, est R² {dec(self.imb['xgb_r2'])} "
                 f"et {eur(self.imb['xgb_mae'])}.")
        if re.search(r"neurone|deep|reseau|lightgbm|catboost|svm|knn", ent.get("q", "")):
            texte += (" Les autres familles (réseaux de neurones, SVM, plus proches voisins) n'ont pas été testées : avec 1 163 annonces "
                      "et des variables tabulaires, les modèles à base d'arbres sont la référence, et l'écart entre les trois "
                      "meilleurs est déjà faible.")
        return Reponse(texte, "comparaison", t, "artefacts/metriques.json.")

    def r_importance(self, ent):
        pi = pd.Series(self.M["permutation"]).sort_values(ascending=False)
        top = pi.head(8)
        t = pd.DataFrame({"Variable": [P.LIBELLES[k] for k in top.index], "Perte de R² si on la brouille": top.map(lambda v: dec(v)).values})
        texte = (f"D'après la permutation importance du modèle XGBoost, les variables les plus utiles sont le nombre de salles de bain "
                 f"(perte de R² de {dec(pi['baths'])} quand on la brouille), le nombre de chambres ({dec(pi['bedrooms'])}) et la piscine "
                 f"({dec(pi['pool'])}), puis le lave-vaisselle, le statut Superhost et le mot « villa » dans le titre. La commune pèse très peu "
                 f"({dec(pi['locality_Saint-François'])} pour Saint-François). Cette mesure dit ce dont le modèle a besoin pour prédire, "
                 "pas l'effet en euros d'une variable : pour cela, voir les coefficients Ridge.")
        return Reponse(texte, "importance", t, "Permutation importance calculée sur les plis de validation (artefacts/metriques.json).")

    def r_limites(self, ent):
        q = pd.DataFrame(self.M["quintiles"])
        t = pd.DataFrame({"Tranche de revenu réel": ["20 % les plus bas", "20 % suivants", "20 % du milieu", "20 % suivants", "20 % les plus hauts"],
                          "Revenu réel moyen": q.reel_moyen.map(eur), "Biais moyen": q.biais_xgb.map(lambda v: eur(v, True)),
                          "Erreur moyenne": q.mae_xgb.map(eur)})
        texte = (f"Le modèle surestime les petits revenus ({eur(q.biais_xgb.iloc[0], True)} en moyenne sur les 20 % les plus bas) et sous-estime "
                 f"les plus élevés ({eur(q.biais_xgb.iloc[-1], True)} sur les 20 % les plus hauts) : c'est la régression vers la moyenne. "
                 "Autres limites : les données sont observationnelles (des associations, pas des causes) ; seuls des logements entiers déjà actifs "
                 "sont observés ; le prix par nuit, la saisonnalité, les notes, les photos et la localisation fine manquent ; le revenu n'est pas la "
                 f"rentabilité, faute de coûts. Pour un bien jamais loué, sans Superhost ni avis, le R² reste de {dec(self.M['bien_neuf']['r2'])}.")
        return Reponse(texte, "limites", t, "Erreurs hors échantillon par quintile de revenu réel (artefacts/metriques.json).")

    def r_donnees(self, ent):
        j = self.M["journal_nettoyage"]
        t = pd.DataFrame(j, columns=["Étape", "Annonces restantes"])
        t["Retirées"] = (-t["Annonces restantes"].diff()).fillna(0).astype(int)
        texte = (f"Les données viennent d'un export AirROI de {entier(j[0][1])} annonces de logements entiers sur Le Gosier, Sainte-Anne et "
                 f"Saint-François. La cible est le revenu des 12 derniers mois (ttm_revenue). Quatre étapes de nettoyage traçables "
                 f"laissent {entier(j[-1][1])} annonces : lignes corrompues, revenu ou avis manquant, annonces inactives ou trop récentes "
                 f"(moins de 2 000 € ou moins de 5 avis) et valeurs extrêmes au-delà de Q3 + 3 × IQR (82 495 €). Les annonces sans nombre "
                 "de chambres sont recodées en studios (0 chambre).")
        return Reponse(texte, "donnees", t, "Journal de nettoyage du pipeline (identique au notebook).")

    def r_validation(self, ent):
        res = pd.DataFrame(self.M["resultats_cv"]).set_index("modele")
        x, rf = res.loc["XGBoost + titre"], res.loc["Random Forest + titre"]
        texte = ("Chaque modèle est évalué en validation croisée à 5 plis, et les hyperparamètres sont réglés par recherche aléatoire. "
                 "Comme le réglage et l'évaluation utilisent les mêmes plis, le score est légèrement optimiste. La validation croisée "
                 "imbriquée corrige ce biais : le réglage est refait à l'intérieur de chaque pli externe (5 plis × 2 répétitions). "
                 f"Résultat retenu : R² {dec(self.imb['xgb_r2'])} au lieu de {dec(x.r2)}. Côté surapprentissage, XGBoost obtient "
                 f"{dec(x.r2_train)} sur l'entraînement contre {dec(x.r2)} en validation, un écart modéré ; Random Forest monte à "
                 f"{dec(rf.r2_train)} en entraînement pour {dec(rf.r2)} en validation, il mémorise davantage.")
        return Reponse(texte, "validation", None, "artefacts/metriques.json.")

    def r_jev(self, ent):
        texte = ("Jev est un modèle de TypeSafe conçu pour être appelé depuis du code : il ne rédige pas de texte, il répond à des "
                 "questions fermées avec une probabilité calibrée. Dans ce projet, il a lu les titres des annonces (gamme, vue mer, plage, "
                 "clientèle, attrait)")
        if self.JR:
            texte += f" : {entier(self.JR['titres'])} titres en {self.JR['duree_s']} secondes pour {dec(self.JR['cout_usd'], 2)} $"
        texte += (". Dans cet assistant, il sert seulement à comprendre la question et à reconnaître les questions hors sujet. "
                  "Les réponses elles-mêmes sont calculées sur les données et le modèle, jamais inventées. Sans clé API, "
                  "des règles par mots-clés prennent le relais.")
        if self.jev is not None and self.jev.appels:
            texte += f" Depuis l'ouverture de cette page : {self.jev.appels} appel(s), environ {dec(self.jev.cout_usd, 4)} $."
        return Reponse(texte, "jev", None, "artefacts/jev_resultats.json.")

    def r_donnee_absente(self, ent):
        texte = ("Cette information n'est pas dans les données du projet. L'export AirROI utilisé contient le revenu annuel, la taille, "
                 "les équipements, le statut Superhost, le nombre d'avis et la commune, mais pas le prix par nuit, le taux d'occupation, "
                 "la saisonnalité, les notes des voyageurs ni les coûts. Je préfère ne pas inventer de chiffre. Je peux en revanche "
                 "répondre sur le revenu annuel, les équipements, les communes ou le modèle.")
        return Reponse(texte, "donnee_absente", None, "Liste des variables de l'export AirROI.")

    def r_aide(self, ent):
        texte = ("Je réponds à partir des 1 163 annonces nettoyées et du modèle du projet. Côté business : revenus par commune ou par taille, "
                 "effet d'un équipement, estimation d'un logement, conseils au propriétaire, audit d'un titre entre guillemets. "
                 "Côté data : précision du modèle, comparaison des algorithmes, variables importantes, validation croisée, limites, nettoyage. "
                 "Chaque chiffre est calculé, jamais inventé.")
        return Reponse(texte, "aide", None, "")

    def r_hors_perimetre(self, ent, compris=True):
        if compris:
            texte = ("Cette question sort du périmètre du projet. Je réponds uniquement sur les locations saisonnières du Gosier, "
                     "de Sainte-Anne et de Saint-François, et sur le modèle qui estime leur revenu.")
        else:
            texte = ("Je n'ai pas reconnu le sujet de la question. Je réponds sur les revenus des locations saisonnières du Gosier, "
                     "de Sainte-Anne et de Saint-François, leurs équipements et le modèle de prédiction. Essayez par exemple une des questions proposées.")
        return Reponse(texte, "hors_perimetre", None, "")


CIBLES = {"famille": "familles", "couple": "couples", "groupe": "groupes", "non_precise": "non précisée"}

QUESTIONS_TITRE = {
    "standing": {"type": "choice", "instructions": "Quel niveau de gamme ce titre met-il en avant ?",
                 "criteria": {"luxe": "haut de gamme, prestige, luxe, villa d'exception",
                              "confort": "logement de bon confort sans revendiquer le luxe",
                              "simple": "logement simple, petit, économique ou basique"}},
    "vue_mer": {"type": "noul", "instructions": "Le titre met en avant une vue sur la mer, l'océan ou le lagon."},
    "plage": {"type": "noul", "instructions": "Le titre met en avant un accès direct ou une grande proximité avec une plage (à pied, pieds dans l'eau, front de mer)."},
    "cible": {"type": "choice", "instructions": "Quelle clientèle ce titre vise-t-il en priorité ?",
              "criteria": {"famille": "familles, enfants, grandes tribus", "couple": "couples, séjour romantique, lune de miel",
                           "groupe": "groupes d'amis, séminaires, grandes capacités", "non_precise": "le titre ne vise aucune clientèle particulière"}},
    "attrait": {"type": "score", "instructions": "À quel point ce titre donne-t-il envie de réserver, par sa précision et ses arguments concrets ?",
                "criteria": ["très faible : titre vague ou générique", "faible", "moyen", "fort", "très fort : arguments concrets et différenciants"]},
}

QUESTIONS_TYPES = {
    "Business": [
        "Quelle commune rapporte le plus ?",
        "Combien rapporte un T2 à Sainte-Anne avec piscine ?",
        "La piscine vaut-elle le coup ?",
        "Quels équipements rapportent le plus ?",
        "J'ai un studio au Gosier sans piscine, que faire pour gagner plus ?",
        "Analyse le titre « Villa vue mer avec piscine privée à Saint-François »",
    ],
    "Data": [
        "Quelle est la précision du modèle ?",
        "Pourquoi XGBoost plutôt qu'un autre modèle ?",
        "Quelles variables comptent le plus ?",
        "Comment avez-vous évité le surapprentissage ?",
        "Quelles sont les limites du modèle ?",
        "Quel prix par nuit faut-il fixer ?",
    ],
}
