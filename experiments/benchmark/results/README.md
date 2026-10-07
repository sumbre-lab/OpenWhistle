# Index des résultats du benchmark

## Résumés à consulter

| Fichier | Usage |
|---|---|
| [rebuttal_pretraining.md](rebuttal_pretraining.md) | Les cinq checkpoints préentraînés du rebuttal, classification sur dix classes et détection |
| [hf_collections_benchmark.csv](hf_collections_benchmark.csv) | Source du tableau, résultats enregistrés du pipeline v5 ; permet aussi de régénérer la collection complète |
| [rebuttal_linear_probing_macro_f1.csv](rebuttal_linear_probing_macro_f1.csv) | Baselines en macro-F1 ; écarts avec certains rapports historiques encore à consolider |
| [watkins_macro_f1/summary.csv](watkins_macro_f1/summary.csv) | Recalcul Watkins ; voir [son protocole](../WATKINS.md) |

Les `.tex` sont les exports LaTeX des tableaux. Aucun entraînement GPU complet
n'a été relancé lors de leur consolidation.

## Rapports servant aux vérifications

- `watkins_macro_f1/` : prédictions, splits, sweeps, bootstrap et provenance.
- `roc_curves/` : figure comparant les sept modèles et les CSV/JSON nécessaires à sa reconstruction.
- `*_detection_benchmark.csv` : rapports de détection des baselines ; certains
  n'incluent pas de bootstrap. Ne pas prendre un écart-type nul pour une
  estimation d'incertitude validée.

## Historique

Les anciens rapports issus des branches et de la soumission initiale sont
conservés dans une archive locale, hors dépôt. Voir
[l'inventaire et les règles d'organisation](../../../docs/results-organization.md).
Cela inclut les rapports texte, `linear_parts/`, les anciennes PR, la figure
UMAP et les tableaux de la collection complète. Les scripts pour les produire
restent disponibles.

Les nouvelles exécutions peuvent écrire `results_classification.txt`,
`results_detection.txt`, `results_openset.txt` et la figure UMAP dans ce dossier.
Ces sorties doivent être examinées avant de remplacer une référence.

Le rapport historique `rebuttal_linear_probing.csv` est également archivé.
Il contient des accuracies, qui ne doivent pas être utilisées sous un en-tête
macro-F1. Les résultats de classification encore à consolider sont indiqués
explicitement dans la page d'entrée du dépôt.
