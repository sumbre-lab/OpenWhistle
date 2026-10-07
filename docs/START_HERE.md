# Où commencer dans OpenWhistle ?

Pour travailler sur le camera-ready, commencer ici. Les rapports détaillés
servent à vérifier les résultats ; il n'est pas nécessaire de les lire tous.

## Le code : quatre blocs

| Ce que tu veux faire | Dossier | Point d'entrée |
|---|---|---|
| Préentraîner Wav2Vec2 | `experiments/pretraining/` | [README et lancement](../experiments/pretraining/README.md) |
| Évaluer des représentations gelées | `experiments/benchmark/` | [README du benchmark](../experiments/benchmark/README.md) |
| Entraîner les encodeurs avec les labels | `experiments/finetuning/` | [README du fine-tuning](../experiments/finetuning/README.md) |
| Détecter et découper les sifflements | `cnn/` | [README du CNN](../cnn/README.md) |

Dans les expériences, `src/` contient les implémentations, les scripts
`run_*.sh` lancent les expériences et `results/` conserve leurs rapports.
`datasets_figures/` sert aux figures descriptives du dataset.

## Les résultats : commencer par les résumés

| Expérience | Fichier à ouvrir | État |
|---|---|---|
| Préentraînement AVES et Wav2Vec2 10/50/100 % | [Table du rebuttal](../experiments/benchmark/results/rebuttal_pretraining.md) | Valeurs retrouvées dans les rapports sauvegardés ; checkpoints identifiés |
| Source des scores de préentraînement | [CSV de référence](../experiments/benchmark/results/hf_collections_benchmark.csv) | Données permettant de reconstruire les tableaux |
| Fine-tuning | [Table des résultats](../experiments/finetuning/results/REBUTTAL_RESULTS.md) | Quatre modèles, moyennes et écarts-types entre seeds |
| Watkins | [Résultats et protocole](../experiments/benchmark/WATKINS.md) | Probe recalculé sur embeddings locaux ; extraction audio à vérifier |
| CNN, quantité de données | [Résumé de la courbe](../cnn/rebuttal/results/learning_curve_summary.csv) | Moyennes des trois répétitions sauvegardées |

La table principale du manuscrit reste à consolider : les anciens scores
en accuracy et les scores en macro-F1 n'utilisent pas tous le même protocole.
Un fichier sauvegardé n'est pas automatiquement un résultat final à publier.

Les métriques par seed, prédictions et données nécessaires aux figures retenues
restent dans leurs sous-dossiers. Les diagnostics par session, figures par run
et rapports historiques ont été déplacés dans une archive locale. Voir
l'[organisation des résultats](results-organization.md) et
l'[index du benchmark](../experiments/benchmark/results/README.md).

## Les fichiers locaux et les sorties de nouvelles expériences

Les checkpoints, caches d'embeddings, fichiers audio et logs volumineux restent
hors Git. Utiliser un dossier de sortie propre pour chaque nouvelle expérience
et conserver la configuration et les versions avec les métriques.
Pour les probes, passer `--results_csv` vers un nouveau fichier évite de
remplacer le CSV de référence ; pour le fine-tuning, choisir `--output_dir`.

L'accord inter-annotateurs est organisé séparément sur le PC et reste hors du
dépôt selon la demande de Pablo.

## Pour vérifier la provenance

- [Préentraînement sur Jean Zay](jeanzay-pretraining-verification.md) : poids
  correspondants, 191k étapes réalisées et pipeline Fairseq des ablations absent.
- [Correspondance checkpoints–tableaux](../experiments/benchmark/PRETRAINING.md).
- [Audit de reproductibilité](reproducibility-audit.md) : ce qu'il reste à fournir.
- [Revue des branches](camera-ready-review.md) : origine des imports.

Les changements camera-ready se trouvent dans la copie de travail
`/tmp/OpenWhistle-camera-ready`. Le dossier initial
`/home/pablo/Documents/OpenWhistle` conserve ses modifications et fichiers locaux.
