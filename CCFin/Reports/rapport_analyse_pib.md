# Rapport d'Analyse Approfondie
## Étude des Demandes de Cartes de Crédit

---

## Contexte Général et Thématique

### Le secteur bancaire à l'ère des données

Dans un environnement économique de plus en plus compétitif et réglementé, les institutions financières font face à un défi majeur : **optimiser leurs processus de décision de crédit tout en gérant efficacement les risques**. L'octroi de cartes de crédit représente un enjeu stratégique crucial pour les banques, car il constitue à la fois une source importante de revenus et un vecteur de risque financier significatif.

### L'importance de l'analyse des demandes de crédit

Chaque année, des millions de demandes de cartes de crédit sont soumises aux institutions financières à travers le monde. **La qualité de l'évaluation de ces demandes a un impact direct sur la rentabilité des banques** : approuver trop facilement peut entraîner des taux de défaut élevés et des pertes financières considérables, tandis qu'un processus trop restrictif peut faire perdre des clients potentiellement solvables à la concurrence.

Traditionnellement, les décisions d'approbation reposaient principalement sur l'expertise humaine et des règles métier simples. Cependant, **l'avènement de l'analyse de données et du machine learning** a révolutionné ce domaine, permettant :

- **Une évaluation plus objective et cohérente** des demandeurs
- **L'identification de patterns complexes** dans les données qui échappent à l'analyse humaine
- **La réduction des biais** dans le processus de décision
- **L'automatisation** des décisions pour les cas standards
- **Une meilleure gestion du risque** à l'échelle du portefeuille

### Problématique de confidentialité et anonymisation

Le secteur bancaire est soumis à des réglementations strictes concernant la protection des données personnelles (RGPD en Europe, diverses lois nationales ailleurs). Dans ce contexte, **l'anonymisation des données** devient une pratique indispensable lorsqu'il s'agit de partager des datasets pour la recherche, l'enseignement ou le développement de modèles prédictifs.

Le dataset que nous analysons dans ce rapport illustre parfaitement cette problématique : **toutes les variables ont été anonymisées** (remplacées par des codes A1, A2, etc.) et les valeurs transformées en symboles sans signification apparente. Cette approche garantit la confidentialité totale des informations tout en préservant les relations statistiques entre les variables, permettant ainsi une analyse pertinente.

### Enjeux de l'analyse de données bancaires

L'analyse de données dans le domaine du crédit soulève plusieurs enjeux fondamentaux :

**1. Enjeu économique** : Les décisions de crédit ont un impact direct sur la rentabilité. Une amélioration de 1% du taux de bonne classification peut représenter des millions d'euros de gains ou de pertes évitées.

**2. Enjeu d'équité et de non-discrimination** : Les algorithmes de décision doivent être exempts de biais discriminatoires basés sur le genre, l'origine ethnique ou d'autres caractéristiques protégées. L'analyse des données permet de détecter et corriger ces biais.

**3. Enjeu réglementaire** : Les banques doivent démontrer que leurs processus de décision sont transparents, auditables et conformes aux réglementations (Bâle III, directives européennes, etc.).

**4. Enjeu de qualité des données** : Les décisions ne peuvent être meilleures que les données sur lesquelles elles reposent. La gestion des données manquantes, des outliers et des erreurs est cruciale.

**5. Enjeu d'explicabilité** : Contrairement aux modèles de "boîte noire", les institutions financières doivent pouvoir expliquer les décisions de refus aux clients et aux régulateurs.

### Objectifs de ce rapport

Face à ces enjeux, ce rapport vise à démontrer une **méthodologie rigoureuse d'analyse de données bancaires**, en couvrant l'ensemble du processus analytique :

- **Explorer et comprendre** la structure des données anonymisées
- **Diagnostiquer et traiter** les problèmes de qualité des données
- **Identifier les patterns et corrélations** entre les caractéristiques des demandeurs
- **Visualiser de manière claire** les insights extraits des données
- **Fournir des recommandations actionnables** pour améliorer le processus de décision

Cette analyse s'inscrit dans une démarche de **data science responsable**, où la rigueur méthodologique, la transparence des processus et la prise en compte des enjeux éthiques sont au cœur de l'approche.

### Structure du document

Le rapport est organisé de manière à guider le lecteur à travers toutes les étapes d'une analyse de données professionnelle, de l'exploration initiale jusqu'aux recommandations finales, en passant par des visualisations avancées et une interprétation approfondie des résultats. Chaque section de code est entièrement documentée et expliquée pour assurer la reproductibilité et la compréhension de la démarche.

---

## 1. Introduction et Contexte

### 1.1 Objectif de l'analyse

Ce rapport vise à analyser en profondeur un ensemble de données concernant les demandes de cartes de crédit. L'objectif principal est de comprendre les caractéristiques des demandeurs, identifier les patterns d'approbation/refus, et extraire des insights actionnables pour optimiser le processus de décision de crédit.

**Objectifs spécifiques :**
- Analyser les profils des demandeurs de cartes de crédit
- Identifier les facteurs influençant l'approbation des demandes
- Détecter les patterns et corrélations entre les variables
- Traiter et analyser l'impact des données manquantes
- Fournir des visualisations permettant une prise de décision éclairée

### 1.2 Méthodologie générale employée

Notre approche méthodologique suit un processus structuré en 5 phases :

1. **Exploration des données** : Compréhension de la structure, types de variables, et qualité des données
2. **Nettoyage et prétraitement** : Traitement des valeurs manquantes et des outliers
3. **Analyse descriptive** : Statistiques univariées et bivariées pour chaque variable
4. **Analyse des relations** : Identification des corrélations et patterns entre variables
5. **Visualisation avancée** : Création de graphiques professionnels pour communiquer les résultats

**Approche technique :**
- Utilisation de Python avec pandas, numpy, matplotlib et seaborn
- Application de techniques statistiques robustes
- Validation croisée des résultats
- Documentation exhaustive de chaque étape

### 1.3 Dataset sélectionné et période d'analyse

**Nom du dataset :** Credit Card Approval Dataset

**Caractéristiques :**
- **Type de données** : Demandes de cartes de crédit
- **Confidentialité** : Données anonymisées (attributs et valeurs remplacés par des symboles)
- **Structure** : Mélange d'attributs continus et catégoriels
- **Qualité** : Présence de valeurs manquantes nécessitant un traitement

**Période d'analyse :** Dataset transversal (cross-sectional) sans dimension temporelle explicite

### 1.4 Questions de recherche principales

1. **Quelles sont les caractéristiques démographiques des demandeurs ?**
   - Distribution des variables catégorielles (genre, statut, etc.)
   - Statistiques des variables continues (âge, revenus, etc.)

2. **Quel est le taux d'approbation global ?**
   - Proportion de demandes approuvées vs refusées
   - Évolution selon les différents segments

3. **Quels facteurs influencent le plus l'approbation ?**
   - Variables les plus corrélées avec la décision
   - Seuils critiques identifiables

4. **Quelle est l'ampleur du problème des données manquantes ?**
   - Proportion de valeurs manquantes par variable
   - Patterns des données manquantes (MCAR, MAR, MNAR)
   - Impact potentiel sur l'analyse

5. **Existe-t-il des segments de demandeurs distincts ?**
   - Clusters naturels dans les données
   - Profils types de demandeurs

---

## 2. Description des Données

### 2.1 Source des données

**Origine :** UCI Machine Learning Repository - Credit Approval Dataset

**Référence :** 
- Repository: https://archive.ics.uci.edu/ml/datasets/credit+approval
- Donateur: Confidential source
- Date de mise à disposition: 1987

**Contexte de collecte :**
Les données proviennent d'une institution financière réelle mais ont été complètement anonymisées pour des raisons de confidentialité. Tous les noms d'attributs et valeurs ont été remplacés par des symboles sans signification (A1, A2, ..., A16).

### 2.2 Variables analysées

Le dataset contient **16 attributs** (15 features + 1 variable cible) :

#### Variables Catégorielles (Nominales)

| Variable | Type | Description | Valeurs possibles | Observations |
|----------|------|-------------|-------------------|--------------|
| A1 | Binaire | Attribut catégoriel #1 | a, b | Genre probable |
| A4 | Catégorielle | Attribut catégoriel #4 | u, y, l, t, autres | Type d'emploi probable |
| A5 | Catégorielle | Attribut catégoriel #5 | g, p, gg, autres | Statut probable |
| A6 | Catégorielle | Attribut catégoriel #6 | c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff | Occupation probable |
| A7 | Catégorielle | Attribut catégoriel #7 | v, h, bb, j, n, z, dd, ff, o | Type de logement probable |
| A9 | Binaire | Attribut catégoriel #9 | t, f | Indicateur booléen |
| A10 | Binaire | Attribut catégoriel #10 | t, f | Indicateur booléen |
| A12 | Binaire | Attribut catégoriel #12 | t, f | Indicateur booléen |
| A13 | Catégorielle | Attribut catégoriel #13 | g, p, s | Catégorie de crédit |

#### Variables Continues (Numériques)

| Variable | Type | Description | Unité | Plage attendue |
|----------|------|-------------|-------|----------------|
| A2 | Continue | Attribut numérique #2 | Années | 13.75 - 80.25 |
| A3 | Continue | Attribut numérique #3 | Montant | 0 - 28 |
| A8 | Continue | Attribut numérique #8 | Montant | 0 - 28.5 |
| A11 | Entier | Attribut numérique #11 | Compteur | 0 - 67 |
| A14 | Continue | Attribut numérique #14 | Montant | 0 - 2000 |
| A15 | Entier | Attribut numérique #15 | Montant | 0 - 100000 |

#### Variable Cible

| Variable | Type | Description | Valeurs |
|----------|------|-------------|---------|
| A16 | Binaire | Décision d'approbation | + (approuvé), - (refusé) |

**Note importante :** En raison de l'anonymisation, nous utiliserons des hypothèses raisonnables sur la signification des variables basées sur le contexte bancaire typique.

### 2.3 Période couverte

**Type de dataset :** Transversal (cross-sectional)

- **Nature temporelle :** Pas de dimension temporelle explicite
- **Date de collecte :** Années 1980
- **Nombre d'observations :** 690 demandes de cartes de crédit
- **Représentativité :** Échantillon représentatif des demandes d'une institution financière

### 2.4 Qualité et limitations des données

#### Points forts

✓ **Taille adéquate** : 690 observations permettent des analyses statistiques fiables

✓ **Variété des types de données** : Mélange équilibré de variables continues et catégorielles

✓ **Variable cible binaire claire** : Approuvé (+) ou Refusé (-) sans ambiguïté

✓ **Données réelles** : Provenant d'une vraie institution financière (crédibilité élevée)

✓ **Anonymisation complète** : Respect total de la confidentialité

#### Limitations identifiées

⚠ **Anonymisation complète** : Impossibilité d'interpréter directement la signification des variables
- Impact : Nécessite des hypothèses et limite l'interprétation business
- Mitigation : Utilisation de l'analyse statistique et du contexte bancaire

⚠ **Valeurs manquantes** : Présence confirmée de données manquantes
- Impact : Potentiel biais dans l'analyse si non traité correctement
- Mitigation : Analyse détaillée des patterns de données manquantes et imputation appropriée

⚠ **Ancienneté des données** : Collectées dans les années 1980
- Impact : Possibles différences avec les pratiques bancaires modernes
- Mitigation : Utilisation principalement pour l'apprentissage méthodologique

⚠ **Absence de contexte temporel** : Pas d'information sur la date des demandes
- Impact : Impossible d'analyser les tendances temporelles
- Mitigation : Focus sur l'analyse transversale

⚠ **Déséquilibre potentiel des classes** : Proportion inconnue a priori entre approuvés/refusés
- Impact : Peut affecter certaines analyses statistiques
- Mitigation : Calcul préliminaire des proportions et ajustement si nécessaire

#### Hypothèses de travail

Basées sur le contexte bancaire standard, nous supposons que :
- A2 pourrait représenter l'âge du demandeur
- A3, A8, A14, A15 pourraient représenter des montants financiers (revenus, dettes, patrimoine)
- A11 pourrait représenter une durée ou un compteur (mois d'emploi, nombre de crédits, etc.)
- A1 pourrait représenter le genre
- A4-A7, A13 représentent probablement des catégories socio-professionnelles

### 2.5 Tableau récapitulatif des données

#### Vue d'ensemble du dataset

| Caractéristique | Valeur | Détails |
|-----------------|--------|---------|
| **Nombre d'observations** | 690 | Demandes de cartes de crédit |
| **Nombre de variables** | 16 | 15 features + 1 variable cible |
| **Variables continues** | 6 | A2, A3, A8, A11, A14, A15 |
| **Variables catégorielles** | 9 | A1, A4, A5, A6, A7, A9, A10, A12, A13 |
| **Variable cible** | 1 | A16 (+ / -) |
| **Valeurs manquantes** | Oui | À quantifier lors du chargement |
| **Taille du fichier** | ~50 KB | Format CSV |

#### Statistiques préliminaires attendues

**Variables continues - Plages attendues :**
- A2 (âge présumé) : 13-80 ans
- A3 : 0-28 unités
- A8 : 0-28.5 unités  
- A11 : 0-67 unités
- A14 : 0-2000 unités
- A15 : 0-100000 unités

**Variables catégorielles - Cardinalité :**
- Binaires (A1, A9, A10, A12) : 2 valeurs chacune
- Faible cardinalité (A4, A5, A13) : 3-5 valeurs
- Haute cardinalité (A6, A7) : 8-14 valeurs

**Variable cible :**
- '+' : Demande approuvée
- '-' : Demande refusée
- Proportion à déterminer lors de l'analyse

---

## 3. Code d'Analyse (Python)

### 3.1 Configuration de l'environnement

Avant de commencer l'analyse, nous devons importer les bibliothèques nécessaires et configurer l'environnement Python pour garantir des visualisations de haute qualité et une manipulation efficace des données.

```python
# ============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT D'ANALYSE
# ============================================================================

# Importation des bibliothèques pour la manipulation de données
import pandas as pd              # Manipulation et analyse de données tabulaires
import numpy as np               # Calculs numériques et opérations sur des arrays

# Importation des bibliothèques pour la visualisation
import matplotlib.pyplot as plt  # Création de graphiques de base
import seaborn as sns           # Visualisations statistiques avancées

# Importation de bibliothèques complémentaires
from datetime import datetime   # Manipulation de dates
import warnings                 # Gestion des avertissements
from scipy import stats        # Tests statistiques

# Configuration de l'affichage des graphiques
plt.style.use('seaborn-v0_8-whitegrid')  # Style professionnel pour les graphiques
sns.set_palette("Set2")                   # Palette de couleurs professionnelle

# Configuration de la taille par défaut des figures
plt.rcParams['figure.figsize'] = (14, 8)  # Largeur: 14 pouces, Hauteur: 8 pouces
plt.rcParams['font.size'] = 11            # Taille de police par défaut
plt.rcParams['axes.labelsize'] = 12       # Taille des labels d'axes
plt.rcParams['axes.titlesize'] = 14       # Taille des titres
plt.rcParams['xtick.labelsize'] = 10      # Taille des ticks X
plt.rcParams['ytick.labelsize'] = 10      # Taille des ticks Y
plt.rcParams['legend.fontsize'] = 10      # Taille de la légende

# Suppression des avertissements non critiques pour une sortie propre
warnings.filterwarnings('ignore')

# Configuration de l'affichage pandas
pd.set_option('display.max_columns', None)          # Afficher toutes les colonnes
pd.set_option('display.max_rows', 100)              # Afficher jusqu'à 100 lignes
pd.set_option('display.precision', 2)               # 2 décimales pour les nombres
pd.set_option('display.float_format', '{:.2f}'.format)  # Format des floats
pd.set_option('display.width', 120)                 # Largeur d'affichage

# Affichage des informations de configuration
print("="*80)
print("CONFIGURATION DE L'ENVIRONNEMENT D'ANALYSE")
print("="*80)
print(f"✓ Python: {sys.version.split()[0]}")
print(f"✓ pandas: {pd.__version__}")
print(f"✓ numpy: {np.__version__}")
print(f"✓ matplotlib: {matplotlib.__version__}")
print(f"✓ seaborn: {sns.__version__}")
print(f"✓ Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ Environnement configuré avec succès\n")
```

**Explication :** Ce bloc initialise toutes les bibliothèques nécessaires et configure les paramètres d'affichage pour garantir des résultats cohérents et professionnels. La configuration est optimisée pour l'analyse de données financières.

---

### 3.2 Chargement et exploration initiale des données

Nous allons maintenant charger le dataset et effectuer une première exploration pour comprendre sa structure.

```python
# ============================================================================
# CHARGEMENT ET EXPLORATION INITIALE DES DONNÉES
# ============================================================================

print("\n" + "="*80)
print("CHARGEMENT DES DONNÉES")
print("="*80)

# Définition des noms des colonnes (A1 à A16)
# Les noms sont anonymisés dans le dataset original
column_names = [f'A{i}' for i in range(1, 17)]

# Chargement du fichier CSV
# Note: Le fichier utilise '?' comme indicateur de valeur manquante
try:
    df = pd.read_csv('credit_approval.csv', 
                     names=column_names,        # Noms des colonnes
                     na_values='?',             # Marqueur de valeurs manquantes
                     skipinitialspace=True)     # Supprimer les espaces initiaux
    print(f"✓ Fichier chargé avec succès")
    print(f"✓ Dimensions du dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes")
except FileNotFoundError:
    print("⚠ Fichier non trouvé. Création d'un dataset synthétique pour démonstration...")
    # Création d'un dataset synthétique pour la démonstration
    np.random.seed(42)
    n_samples = 690
    
    df = pd.DataFrame({
        'A1': np.random.choice(['a', 'b'], n_samples),
        'A2': np.random.uniform(15, 75, n_samples),
        'A3': np.random.uniform(0, 20, n_samples),
        'A4': np.random.choice(['u', 'y', 'l', 't'], n_samples),
        'A5': np.random.choice(['g', 'p', 'gg'], n_samples),
        'A6': np.random.choice(['c', 'w', 'q', 'k', 'i', 'aa', 'ff', 'x'], n_samples),
        'A7': np.random.choice(['v', 'h', 'bb', 'z', 'ff'], n_samples),
        'A8': np.random.uniform(0, 15, n_samples),
        'A9': np.random.choice(['t', 'f'], n_samples),
        'A10': np.random.choice(['t', 'f'], n_samples),
        'A11': np.random.randint(0, 30, n_samples),
        'A12': np.random.choice(['t', 'f'], n_samples),
        'A13': np.random.choice(['g', 'p', 's'], n_samples),
        'A14': np.random.uniform(0, 1000, n_samples),
        'A15': np.random.randint(0, 50000, n_samples),
        'A16': np.random.choice(['+', '-'], n_samples, p=[0.55, 0.45])
    })
    
    # Introduction aléatoire de valeurs manquantes (5% du dataset)
    for col in df.columns[:-1]:  # Sauf A16 (variable cible)
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    print(f"✓ Dataset synthétique créé: {df.shape[0]} lignes × {df.shape[1]} colonnes")

# Affichage des premières lignes
print("\n" + "-"*80)
print("APERÇU DES PREMIÈRES LIGNES DU DATASET")
print("-"*80)
print(df.head(10))

# Affichage des dernières lignes pour vérifier la cohérence
print("\n" + "-"*80)
print("APERÇU DES DERNIÈRES LIGNES DU DATASET")
print("-"*80)
print(df.tail(5))

# Informations générales sur le dataset
print("\n" + "-"*80)
print("INFORMATIONS GÉNÉRALES SUR LE DATASET")
print("-"*80)
print(df.info())

# Types de données par colonne
print("\n" + "-"*80)
print("TYPES DE DONNÉES")
print("-"*80)
print(df.dtypes)

print("\n✓ Exploration initiale terminée")
```

**Explication :** Ce code charge le dataset et affiche des informations essentielles sur sa structure. Si le fichier n'est pas disponible, un dataset synthétique est créé pour démonstration. L'utilisation de `na_values='?'` permet de gérer correctement les valeurs manquantes dans le format original.

---

### 3.3 Identification et conversion des types de données

Les données sont souvent chargées avec des types incorrects. Nous devons identifier et convertir correctement chaque variable.

```python
# ============================================================================
# IDENTIFICATION ET CONVERSION DES TYPES DE DONNÉES
# ============================================================================

print("\n" + "="*80)
print("IDENTIFICATION DES TYPES DE VARIABLES")
print("="*80)

# Définition manuelle des types de variables basée sur la documentation
variables_continues = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
variables_categorielles = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
variable_cible = 'A16'

print(f"\n✓ Variables continues: {len(variables_continues)}")
print(f"  {', '.join(variables_continues)}")

print(f"\n✓ Variables catégorielles: {len(variables_categorielles)}")
print(f"  {', '.join(variables_categorielles)}")

print(f"\n✓ Variable cible: {variable_cible}")

# Conversion des types de données
print("\n" + "-"*80)
print("CONVERSION DES TYPES DE DONNÉES")
print("-"*80)

# Conversion des variables continues en float
for col in variables_continues:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"✓ {col}: converti en numérique (float)")
    except Exception as e:
        print(f"⚠ {col}: erreur de conversion - {str(e)}")

# Conversion des variables catégorielles en type 'category'
for col in variables_categorielles + [variable_cible]:
    try:
        df[col] = df[col].astype('category')
        print(f"✓ {col}: converti en catégoriel ({df[col].nunique()} catégories)")
    except Exception as e:
        print(f"⚠ {col}: erreur de conversion - {str(e)}")

# Vérification des conversions
print("\n" + "-"*80)
print("TYPES DE DONNÉES APRÈS CONVERSION")
print("-"*80)
print(df.dtypes)

# Résumé des types
print("\n" + "-"*80)
print("RÉSUMÉ DES TYPES DE DONNÉES")
print("-"*80)
print(f"Variables numériques (float64): {(df.dtypes == 'float64').sum()}")
print(f"Variables catégorielles (category): {(df.dtypes == 'category').sum()}")
print(f"Autres types: {((df.dtypes != 'float64') & (df.dtypes != 'category')).sum()}")

print("\n✓ Conversion des types terminée")
```

**Explication :** Ce code identifie et convertit correctement chaque variable selon son type réel. Les variables continues sont converties en `float64` pour les calculs numériques, tandis que les variables catégorielles sont converties en type `category` pour optimiser la mémoire et faciliter les analyses.

---

### 3.4 Analyse des valeurs manquantes

L'analyse des valeurs manquantes est cruciale pour comprendre la qualité des données et décider du traitement approprié.

```python
# ============================================================================
# ANALYSE DÉTAILLÉE DES VALEURS MANQUANTES
# ============================================================================

print("\n" + "="*80)
print("ANALYSE DES VALEURS MANQUANTES")
print("="*80)

# Calcul des statistiques de valeurs manquantes
valeurs_manquantes = df.isnull().sum()
pourcentage_manquant = (valeurs_manquantes / len(df)) * 100

# Création d'un DataFrame récapitulatif
missing_df = pd.DataFrame({
    'Variable': df.columns,
    'Valeurs_manquantes': valeurs_manquantes.values,
    'Pourcentage': pourcentage_manquant.values,
    'Type': df.dtypes.values
}).sort_values('Pourcentage', ascending=False)

# Affichage du tableau
print("\n" + "-"*80)
print("TABLEAU RÉCAPITULATIF DES VALEURS MANQUANTES")
print("-"*80)
print(missing_df.to_string(index=False))

# Statistiques globales
print("\n" + "-"*80)
print("STATISTIQUES GLOBALES")
print("-"*80)
print(f"Nombre total de cellules: {df.size:,}")
print(f"Nombre de cellules avec valeurs manquantes: {df.isnull().sum().sum():,}")
print(f"Pourcentage global de valeurs manquantes: {(df.isnull().sum().sum() / df.size) * 100:.2f}%")

# Identification des variables problématiques (>10% de données manquantes)
variables_problematiques = missing_df[missing_df['Pourcentage'] > 10]
if len(variables_problematiques) > 0:
    print(f"\n⚠ {len(variables_problematiques)} variable(s) avec >10% de données manquantes:")
    for idx, row in variables_problematiques.iterrows():
        print(f"  - {row['Variable']}: {row['Pourcentage']:.2f}%")
else:
    print("\n✓ Aucune variable avec plus de 10% de données manquantes")

# Analyse des lignes avec valeurs manquantes
lignes_completes = df.dropna().shape[0]
lignes_avec_manquantes = df.shape[0] - lignes_completes

print(f"\nLignes complètes (sans valeurs manquantes): {lignes_completes} ({(lignes_completes/len(df))*100:.2f}%)")
print(f"Lignes avec au moins une valeur manquante: {lignes_avec_manquantes} ({(lignes_avec_manquantes/len(df))*100:.2f}%)")

# Distribution du nombre de valeurs manquantes par ligne
missing_per_row = df.isnull().sum(axis=1)
print("\n" + "-"*80)
print("DISTRIBUTION DU NOMBRE DE VALEURS MANQUANTES PAR LIGNE")
print("-"*80)
print(missing_per_row.value_counts().sort_index())

# Test de pattern de données manquantes (MCAR, MAR, MNAR)
print("\n" + "-"*80)
print("ANALYSE DES PATTERNS DE DONNÉES MANQUANTES")
print("-"*80)

# Corrélation entre les données manquantes
missing_matrix = df.isnull().astype(int)
missing_corr = missing_matrix.corr()

# Identifier les paires de variables avec forte corrélation de manquance
high_corr_missing = []
for i in range(len(missing_corr.columns)):
    for j in range(i+1, len(missing_corr.columns)):
        if abs(missing_corr.iloc[i, j]) > 0.3 and missing_corr.iloc[i, j] != 1:
            high_corr_missing.append({
                'Var1': missing_corr.columns[i],
                'Var2': missing_corr.columns[j],
                'Corrélation': missing_corr.iloc[i, j]
            })

if high_corr_missing:
    print("\n⚠ Corrélations significatives détectées entre les patterns de données manquantes:")
    print("   (Suggère que les données ne sont peut-être pas MCAR)")
    for item in high_corr_missing:
        print(f"  - {item['Var1']} ↔ {item['Var2']}: r = {item['Corrélation']:.3f}")
else:
    print("\n✓ Pas de corrélation forte entre les patterns de données manquantes")
    print("   (Compatible avec l'hypothèse MCAR - Missing Completely At Random)")

print("\n✓ Analyse des valeurs manquantes terminée")
```

**Explication :** Ce code effectue une analyse exhaustive des valeurs manquantes, incluant leur distribution, leur proportion, et la détection de patterns. L'analyse des corrélations entre les données manquantes permet d'identifier si les valeurs sont manquantes complètement au hasard (MCAR), manquantes au hasard (MAR), ou manquantes de façon non aléatoire (MNAR).

---

### 3.5 Traitement des valeurs manquantes

Basé sur l'analyse précédente, nous appliquons des stratégies de traitement appropriées.

```python
# ============================================================================
# TRAITEMENT DES VALEURS MANQUANTES
# ============================================================================

print("\n" + "="*80)
print("TRAITEMENT DES VALEURS MANQUANTES")
print("="*80)

# Création d'une copie du DataFrame pour le traitement
df_cleaned = df.copy()

print("\n" + "-"*80)
print("STRATÉGIES DE TRAITEMENT PAR TYPE DE VARIABLE")
print("-"*80)

# 1. TRAITEMENT DES VARIABLES CONTINUES
print("\n1. Variables continues - Imputation par la médiane")
print("   (Plus robuste aux outliers que la moyenne)")

for col in variables_continues:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        # Calcul de la médiane avant imputation
        mediane = df_cleaned[col].median()
        # Imputation
        df_cleaned[col].fillna(mediane, inplace=True)
        print(f"   ✓ {col}: {missing_count} valeurs imputées avec la médiane ({mediane:.2f})")
    else:
        print(f"   ✓ {col}: aucune valeur manquante")

# 2. TRAITEMENT DES VARIABLES CATÉGORIELLES
print("\n2. Variables catégorielles - Imputation par le mode")
print("   (Valeur la plus fréquente)")

for col in variables_categorielles:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        # Calcul du mode avant imputation
        mode_value = df_cleaned[col].mode()[0]
        # Imputation
        df_cleaned[col].fillna(mode_value, inplace=True)
        print(f"   ✓ {col}: {missing_count} valeurs imputées avec le mode ('{mode_value}')")
    else:
        print(f"   ✓ {col}: aucune valeur manquante")

# 3. VÉRIFICATION DE LA VARIABLE C