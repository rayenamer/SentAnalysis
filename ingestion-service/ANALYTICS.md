# POST /ingest/analyze : agrégations et signaux

Les champs existants ingestion et analysis sont conservés. Trois champs s'ajoutent :
aggregation, weighted_aggregation et signals. Tous les calculs sont effectués
par les endpoints HTTP existants d'Insight. Les modèles VADER, FinBERT et
loughran_mcdonald suivent le même parcours.

Chaque étape renvoie status (success, unavailable ou error), reason et data.
Une erreur d'une étape ne supprime pas les documents ni les scores précédents.
Les messages d'erreur libres d'Insight ne sont jamais recopiés. Les diagnostics
des réponses valides restent présents, même sans résultat numérique calculable.

## Requête minimale

```json
{"providers":["hackernews"],"model":"loughran_mcdonald","max_documents":3}
```

Les agrégations simples et pondérées portent sur les seuls documents analysés
avec succès dans le lot, sur 1h, 24h et 7d. Elles ne représentent pas tous les
articles de ces périodes. Les dates de publication et tickers sont conservés.
Une date absente ne devient pas la date de collecte. Les exclusions restent
visibles dans diagnostics. Les poids par défaut d'Insight sont utilisés :
poids de source 1, facteur d'engagement 0. Pour Hacker News, num_comments est
transmis comme engagement s'il est valide.

Une correspondance entreprise/ticker doit être configurée, ou le texte doit
contenir un cashtag. Les articles sans ticker ne créent pas artificiellement d'actif.

## Historique optionnel

L'objet analytics accepte :

- signal_window : 1h, 24h (défaut), 7d ;
- history : au plus 1000 observations d'agrégats comparables ;
- prices : au plus 2000 clôtures réelles ;
- time_bucket : hourly (défaut), daily.

Une observation history contient ticker, model, sentiment_score, observed_at,
aggregation_window (24h par défaut), score_kind (raw uniquement). Modèle et
horizon doivent correspondre à la requête. Scores finis dans [-1,1], dates
avec fuseau, tickers normalisés. Données futures et doublons contradictoires
sont rejetés en 422.

Un prix contient ticker, timestamp et close_price strictement positif et fini.
Les clôtures sont exactement alignées en UTC : heure exacte pour hourly,
minuit pour daily. Si prices est fourni, history doit respecter cet alignement.
Aucun prix n'est téléchargé automatiquement.

L'agrégat courant est observé à sa date réelle de calcul, après analyse.
Il complète history pour vélocité et anomalies. Une observation unique ne
suffit pas à calculer une vélocité. Les anomalies demandent au moins cinq
points historiques et une variance non nulle, selon les paramètres Insight.
Le premier résultat d'une série peut rester nul même si les suivants sont calculés.

Corrélation et Granger utilisent uniquement les agrégats et prix exactement
alignés. L'agrégat courant n'est jamais arrondi ni antidaté pour rentrer dans
une clôture passée. Sans prix : prices_not_provided. Sans historique aligné :
no_bucket_aligned_sentiment_history. Quand Insight reçoit une série insuffisante,
ses diagnostics sont conservés dans data. Les paramètres statistiques restent
les valeurs par défaut des APIs Insight.

La plateforme hôte conserve l'historique ; Ingestion ne stocke rien entre appels.
Pour réutiliser un agrégat réel, prendre mean_sentiment du ticker dans la fenêtre
choisie, et as_of comme observed_at. Ne pas utiliser la publication d'un article
comme date de disponibilité de l'agrégat.

## Exécution et vérification

Les appels sont séquentiels et bornés par les timeout et limites de réponse
Insight existants, sans relance automatique des étapes analytiques. Le temps
total peut atteindre la somme de ces délais. Aucun nouveau modèle ni paquet.

Tests automatiques : contrats, erreurs partielles, réponses incohérentes, absence
de date/ticker, validation d'historique, absence d'antidatage, deux serveurs Uvicorn
réels avec source RSS et historique/prix de test synthétiques.

Essai direct du 30 octobre 2026 : Hacker News → Ingestion (8001) → Insight (8000),
deux appels HTTP 200, trois documents par appel, aucune erreur d'analyse.
Apple → AAPL appliqué au processus uniquement, .env inchangé.
Moyenne AAPL sur 24h et 7d : 1/3, trois articles (un positif, deux neutres).
Fenêtre 1h vide, conformément aux dates récupérées. Le second appel a réutilisé
le véritable agrégat du premier : vélocité 0, direction stable.
Anomalies : insufficient_history. Corrélation et Granger : prices_not_provided.
Ces derniers signaux ne sont pas présentés comme calculés sur des données réelles.
Aucun historique ni prix financier n'a été inventé pour cet essai.
