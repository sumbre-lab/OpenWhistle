# Organisation des résultats

Le 7 octobre 2026, Pablo a demandé de garder les résultats utiles au manuscrit
dans la branche camera-ready et d'archiver le reste en local.

## Ce qui reste dans le dépôt

- **Benchmark** : tableau de préentraînement du rebuttal et CSV source,
  baselines macro-F1/détection à consolider, figure ROC comparative et sources,
  résultats Watkins avec prédictions, splits, sweep et provenance.
- **Fine-tuning** : résumé et métriques JSON des modèles/seeds du rebuttal,
  avec les configurations équilibrées et complètes pour les comparaisons.
- **Courbe CNN** : figure agrégée, CSV de synthèse, protocole, métriques et
  metadata des 15 runs. Les métriques par seed permettent de recalculer les ±.
- **Figures du dataset** : figures agrégées et leurs données de préparation.

Le rangement n'établit pas une validation scientifique supplémentaire :
les écarts de protocoles décrits dans l'audit restent à résoudre.

## Archive locale

Emplacement permanent sur le PC :

```text
/home/pablo/Documents/OpenWhistle-archives/camera-ready-2026-10-07/
```

177 fichiers, environ 13,6 MiB, ont été copiés et vérifiés par SHA-256 avant
leur retrait de la copie camera-ready :

| Groupe | Fichiers archivés | Contenu |
|---|---:|---|
| Benchmark | 47 | Anciens rapports, exports intermédiaires, PR historiques, UMAP, plots ROC individuels/doublons, sorties bootstrap Watkins régénérables |
| Fine-tuning | 10 | Runs supplémentaires de la collection, hors résumé du rebuttal |
| CNN | 120 | Figures de chaque run, matrices et diagnostics par session |

Les chemins relatifs d'origine sont conservés dans l'archive. Son
`manifest.json` donne les hashes, tailles, raisons et commit source.
L'[inventaire suivi dans Git](archived-results.csv) permet de retrouver chaque
fichier sans remettre tout le contenu dans le dépôt.

Pour récupérer un fichier, copier son chemin depuis l'archive vers un dossier
de travail dédié. Pour examiner tous les résultats dans Git, le commit
`0b1a70a` conserve l'état précédant le rangement.

Le dossier de travail initial `/home/pablo/Documents/OpenWhistle` et ses
modifications non commitées n'ont pas été nettoyés. Les poids et gros caches
qui s'y trouvent restent à leur emplacement d'origine.
