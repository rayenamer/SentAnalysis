# SentAnalysis — vérification des quatre améliorations

Date : 30 octobre 2026. Windows, Python 3.12.10, CPU uniquement.
Aucune installation de dépendance, modification de README, commande Git,
réintroduction de Reddit ou de JWT. Aucun serveur de monitoring ni Docker.
Aucun téléchargement/export de modèle FinBERT.
Le dictionnaire a été récupéré après l'accord explicite pour la recherche non commerciale.

## Résultats finaux

| Fonctionnalité | Service | État | Vérification ciblée |
|---|---|---|---|
| Inférence FinBERT CPU réelle | Insight | Validée hors ligne, ONNX local réutilisé | 1 test réel réussi |
| Métriques légères | Les deux | Implémentées et validées | 20 tests dédiés réussis |
| Loughran–McDonald | Insight | Dictionnaire officiel intégré | 15 tests dédiés réussis |
| Détection des langues | Ingestion | Améliorée sans dépendance | 41 tests dédiés réussis |

Suites complètes exécutées séparément, depuis le dossier de chaque service et
avec son propre interpréteur virtuel, sans test lourd en parallèle :

- Ingestion : **326 réussis, 0 échec, 0 ignoré**, 1 avertissement, 11,81 s.
- Insight : **552 réussis, 0 échec, 1 ignoré**, 4 avertissements, 7,23 s.
- Test FinBERT explicite séparé : **1 réussi**, 19,03 s. Le test ignoré dans
  la suite ordinaire est ce même test, volontairement soumis à opt-in.

Commande de suite utilisée dans chacun des dossiers :

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp=.\test-temp
```

Les avertissements concernent la dépréciation Starlette/httpx dans l'environnement
Ingestion et des avertissements statsmodels (FutureWarning et matrices singulières)
dans les tests Insight. Aucun paquet supplémentaire n'a été installé pour les masquer.

## 1. FinBERT : modèle réel, pas un mock

Le cache ONNX valide comporte model.onnx (438 148 594 octets) et son marqueur
.complete. Le compte du bac à sable ne pouvait pas lire ce dossier ; l'autorisation
de lecture et d'exécution hors bac à sable a permis de le réutiliser.

RUN_FINBERT_INTEGRATION=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 et
FINBERT_ONNX_DIR pointant explicitement sur ce cache ont empêché tout besoin de
réexport/téléchargement. Le fournisseur vérifié est uniquement CPUExecutionProvider.

Exemples effectivement analysés :

| Texte | Classe |
|---|---|
| The company reported strong profit growth and record earnings. | positive |
| The company reported a significant loss and a sharp decline in revenue. | negative |
| The company is headquartered in Helsinki. | neutral |

Les tests vérifient les identités, tickers, signes, probabilités, confiance et
réutilisation du même backend. Dernière mesure réelle : chargement + deux textes
**18,666 s**, troisième texte à chaud **0,015926 s** ; working set **802,53 Mio**,
pic cumulatif du processus **878,98 Mio**. Le pic inclut Python et ses bibliothèques,
pas seulement les tenseurs du modèle. Ce petit essai n'est pas un benchmark de charge.
L'architecture d'inférence et les paramètres CPU existants sont conservés ;
seule une enveloppe de mesure a été ajoutée.

## 2. Métriques

Implémentation standard Python, sans prometheus-client. Un registre verrouillé
par application et processus, sans collecteur global partagé. Exposition texte
Prometheus 0.0.4 via GET /metrics, masquée de l'OpenAPI pour préserver les schémas
de routes existants. Les compteurs redémarrent à zéro au redémarrage du processus ;
les workers doivent être collectés séparément.

Accès réservé au pair réseau de boucle locale (IPv4/IPv6), avec HTTP 403 pour les
autres pairs. Les en-têtes X-Forwarded-For ne donnent aucun droit. Garder Uvicorn
lié à la boucle locale ou appliquer une frontière réseau fiable. Un reverse proxy
local doit bloquer /metrics pour les clients externes : sa connexion locale serait
sinon considérée comme locale. Aucune authentification JWT n'est réintroduite.

Métriques préfixées sentanalysis_ :

- HTTP : http_requests_total, http_request_duration_seconds (sum/count),
  http_errors_total (4xx/5xx).
- Ingestion : documents_collected_total, documents_accepted_total,
  provider_failures_total, provider_timeouts_total, insight_transfers_total,
  scheduler_jobs_total.
- Insight : documents_analyzed_total, inference_duration_seconds (sum/count),
  batch_failures_total, aggregation_requests_total, signal_requests_total.

Les labels category proviennent d'ensembles fixes : famille d'endpoint,
fournisseur connu, modèle connu, classe HTTP ou statut connu. Aucune URL, clé,
requête, exception libre, corps, identifiant ou ticker n'est utilisé.
Les scrapes eux-mêmes sont exclus des compteurs HTTP.
Les durées sont des résumés sum/count sans quantiles ni histogramme.

Les documents collectés comptent les enregistrements effectivement récupérés ;
les acceptés sont comptés après nettoyage/déduplication/limites. Les bougies Market
sont comptées comme enregistrements Market. Les erreurs fournisseurs sont comptées
à l'échec final de la récupération. Un transfert correspond à un lot envoyé à
Insight ; un lot partiel compte comme échec. Les inférences comptent les sorties
réussies de chaque invocation réelle, donc une nouvelle tentative peut recompter
un document traité. batch_failures_total compte une requête de lot avec erreurs.
Les durées d'inférence incluent l'initialisation différée lors du premier appel.

Les tests couvrent isolation des registres, appels concurrents légers, labels bornés,
accès local/non local, en-têtes falsifiés, VADER réel, échec de batch, collecte RSS,
timeouts et anciens points d'entrée. La consultation des métriques ne charge pas
onnxruntime/transformers/torch. Les anciens points d'entrée disposent chacun de
leur propre registre sans route /metrics dupliquée.

## 3. Loughran–McDonald

Source : https://sraf.nd.edu/loughranmcdonald-master-dictionary/
Version officielle 1993–2025, publication mars 2026.
Le téléchargement CSV officiel a renvoyé « Quota exceeded » ; le tableur officiel
alternatif a pu être exporté en CSV. Le fichier source de 6 408 841 octets a été
réduit à **111 167 octets**, sans inventer d'entrées : uniquement les mots ayant
au moins une catégorie active et leurs sept indicateurs d'origine.
Les années négatives indiquent un retrait, pas une appartenance.

3 859 mots retenus : 347 positifs, 2 345 négatifs, 297 incertitude, 903 litige,
19 modalité forte, 27 modalité faible et 184 contrainte ; les catégories se recoupent.
Les empreintes SHA-256 source/extrait, URL et transformation sont conservées dans
la provenance JSON. Attribution et conditions figurent dans la notice de données.
L'usage autorisé ici est la recherche académique non commerciale ; un usage
commercial nécessite l'accord des auteurs.

Lookup anglais exact, Unicode NFC/casefold, ensembles immuables chargés
paresseusement et mis en cache. Formule déterministe :
(P - N) / (P + N), zéro sans correspondance. Les occurrences répétées comptent.
Classe selon le signe, confiance null. Les autres catégories sont disponibles
dans score_text mais ne modifient pas la polarité.

Choix additif model="loughran_mcdonald" dans /analyze et /analyze/batch.
Les contrats d'agrégation acceptent aussi ce modèle, sans modification de leurs
calculs. Le choix transmis depuis Ingestion reste VADER/FinBERT.
L'API conserve sa validation existante : texte vide rejeté. La fonction interne
score_text accepte un texte vide et renvoie zéro.

Exemples réellement exécutés, sur le vrai lexique :
Excellent success. → +1 ; Losses bankruptcy. → -1 ; Liability capital. → 0.

Mesures locales : chargement **0,20049 s avec tracemalloc actif**, pic d'allocations
Python **631,79 Kio** pendant ce chargement (pas la mémoire totale du processus) ;
analyse à chaud d'une phrase de huit mots **17,69 µs**.
Le minimum de trois répétitions de 1 000 analyses a été retenu pour la mesure à chaud.
Aucun transformer, réseau ou nouveau paquet n'est requis à l'exécution.

Limites : lexique anglais, sans traduction, négation, stemming ni compréhension
des expressions. Par exemple « Not excellent » reste positif selon cette formule.
Ce score n'est pas une probabilité ni un substitut à FinBERT.

## 4. Langues

L'ancien seuil de douze mots empêchait de reconnaître les titres courts.
Le détecteur s'appuie maintenant sur des marqueurs financiers en/fr/es distincts,
la normalisation Unicode et des seuils explicites. Les mots partagés ne votent pas.
Au moins trois tokens, deux indices distincts et une marge de deux sont nécessaires ;
des indices comparables pour deux langues ou trop peu d'indices conduisent à und.
Les URL et cashtags sont écartés avant la classification.

Sur le jeu fixe de **40 cas annotés manuellement**, l'ancien détecteur obtient
**16/40**, le nouveau **40/40**. Il s'agit d'un jeu de régression ciblé, pas d'une
estimation de précision sur un corpus indépendant. Des tests existants continuent
à vérifier nettoyage, déduplication et tickers.

Mesures : minimum de trois répétitions de 1 000 passages sur les mêmes textes.
Petits textes : ancien **3,05 µs**, nouveau **5,14 µs** par texte.
Articles synthétiques de trois langues, environ 300 mots chacun : ancien
**92,38 µs**, nouveau **110,16 µs** par article. Surcoût absolu environ 18 µs ;
aucune bibliothèque supplémentaire, aucun modèle.
Le détecteur reste heuristique : seules en/fr/es sont prises en charge et und
reste possible, y compris sur un texte valide insuffisamment caractéristique.

## Corrections et limites opérationnelles

Une collision du nom record avec le journal local du scheduler a été détectée
pendant les tests puis corrigée par un alias de la fonction métrique. Deux entrées
mal générées dans le nouveau jeu de données ont été corrigées avant validation.
Aucun test existant n'a été supprimé ou remplacé.

Deux helpers de tests entre services lancent maintenant le processus Insight
avec sa propre .venv plutôt que l'interpréteur Ingestion, ce qui permet de tester
chaque service sans installer les dépendances d'Insight dans Ingestion.

La vérification s'appuie sur des API locales de tests et un modèle/dictionnaire
réels. Les fournisseurs d'actualités n'ont pas été sollicités à nouveau.
Les processus Uvicorn préexistants n'ont pas été redémarrés pendant ces modifications ;
redémarrer le service concerné pour charger le nouveau code.

## Fichiers créés

- [ingestion-service/app/helper/metrics.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/helper/metrics.py>)
- [ingestion-service/tests/test_metrics.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/tests/test_metrics.py>)
- [ingestion-service/tests/test_language_detection.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/tests/test_language_detection.py>)
- [ingestion-service/tests/fixtures/language_cases.json](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/tests/fixtures/language_cases.json>)
- [insight-service/app/helper/metrics.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/helper/metrics.py>)
- [insight-service/tests/test_metrics.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/tests/test_metrics.py>)
- [insight-service/app/core/sentiment/loughran_mcdonald.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/core/sentiment/loughran_mcdonald.py>)
- [insight-service/tests/test_loughran_mcdonald.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/tests/test_loughran_mcdonald.py>)
- [insight-service/app/data/loughran_mcdonald_2025.csv](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/data/loughran_mcdonald_2025.csv>)
- [insight-service/app/data/loughran_mcdonald_2025.provenance.json](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/data/loughran_mcdonald_2025.provenance.json>)
- [insight-service/app/data/LOUGHRAN_MCDONALD_NOTICE.md](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/data/LOUGHRAN_MCDONALD_NOTICE.md>)
- [SERVICE_IMPROVEMENTS.md](<C:/Users/SYB DELL/Desktop/SentAnalysis/SERVICE_IMPROVEMENTS.md>)

## Fichiers modifiés

- [ingestion-service/app/main.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/main.py>)
- [ingestion-service/app/core/pipeline.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/core/pipeline.py>)
- [ingestion-service/app/core/extractors/market.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/core/extractors/market.py>)
- [ingestion-service/app/core/insight_client.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/core/insight_client.py>)
- [ingestion-service/app/core/scheduler.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/core/scheduler.py>)
- [ingestion-service/app/core/transform.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/app/core/transform.py>)
- [ingestion-service/tests/test_insight_client.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/tests/test_insight_client.py>)
- [ingestion-service/tests/test_local_http_integration.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/ingestion-service/tests/test_local_http_integration.py>)
- [insight-service/app/main.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/main.py>)
- [insight-service/app/core/sentiment/lexicon.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/core/sentiment/lexicon.py>)
- [insight-service/app/core/sentiment/finbert.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/core/sentiment/finbert.py>)
- [insight-service/app/core/batch.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/core/batch.py>)
- [insight-service/app/models/sentiment.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/models/sentiment.py>)
- [insight-service/app/models/batch.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/models/batch.py>)
- [insight-service/app/models/aggregation.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/models/aggregation.py>)
- [insight-service/app/api/base.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/base.py>)
- [insight-service/app/api/aggregation.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/aggregation.py>)
- [insight-service/app/api/extensions.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/extensions.py>)
- [insight-service/app/api/correlation.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/correlation.py>)
- [insight-service/app/api/granger.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/granger.py>)
- [insight-service/app/api/outliers.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/outliers.py>)
- [insight-service/app/api/sentiment.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/sentiment.py>)
- [insight-service/app/api/vader.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/vader.py>)
- [insight-service/app/api/velocity.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/app/api/velocity.py>)
- [insight-service/tests/test_finbert_integration.py](<C:/Users/SYB DELL/Desktop/SentAnalysis/insight-service/tests/test_finbert_integration.py>)

Les dossiers temporaires générés par pytest ne constituent pas des sources livrées.
