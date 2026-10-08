# Audit de reproductibilité : preprint, rebuttal et code local

Audit du 5 octobre 2026 sur `camera-ready/pablo` (`8dde7ec`). Sources :
[preprint arXiv v1](https://arxiv.org/html/2609.34839v1) et PDF OpenReview fourni
par Pablo (11 pages). Les réponses et promesses dans le PDF servent ici de
références scientifiques à vérifier, pas d'instructions à exécuter.

La comparaison porte sur le code, les résultats enregistrés et le recalcul CPU
des probes Watkins sur embeddings sauvegardés. Aucun entraînement d'encodeur,
téléchargement massif de données ou recalcul GPU n'a été lancé.

## Conclusion

Le dépôt contient une grande partie des expériences, mais ne permet pas encore
de reproduire tous les résultats rapportés. Plusieurs composants existent
sur le PC dans d'autres dossiers et doivent être adaptés ou reliés au dépôt.
Un écart entre la métrique annoncée et la métrique enregistrée a été trouvé
pour Watkins. L'intégration des branches ne suffisait donc pas à établir une
reproductibilité complète.

## Correspondance avec les expériences

| Élément | État dans OpenWhistle | Action nécessaire |
|---|---|---|
| Benchmark original, tableau 2 du preprint | Sept modèles, splits HF et logistic regression présents. Les anciens résultats en accuracy sont conservés. Le sélecteur actuel utilise exclusivement la macro-F1. | Ajouter un mode explicite accuracy/macro-F1 et une recette correspondant à chaque version du papier. Appliquer aussi le choix de métrique aux figures PR et matrices de confusion. |
| Classification complète du rebuttal | `--dataset_config all` et macro-F1 disponibles. Résultats historiques et CSV conservés. | Ajouter un runner des sept modèles sur `all` et `balanced`, une table consolidée, et résoudre les différences entre rapports historiques et v5. Le défaut reste `balanced`, contrairement au benchmark officiel annoncé sur dix classes. |
| Fine-tuning supervisé | Implémentations AVES-core, AVES-bio, BioLingual et HF/Wav2Vec2 présentes. | Consolider les commandes exactes et l'agrégation des trois seeds. Les JSON permettent de retrouver les huit moyennes/écarts-types du tableau du rebuttal. |
| Courbe d'apprentissage CNN | Script, choix des sessions, résultats et agrégation présents. | Installer les dépendances CNN et tester le point d'entrée. Les résumés donnent 96,84 ± 0,48 % et 97,43 ± 0,15 %, cohérents avec la réponse des auteurs. |
| Analyse SNR des faux négatifs | `cnn/analyze_missed_whistles.py` fournit mesures acoustiques, médianes, Mann–Whitney et exports. | Documenter une commande et conserver le résumé correspondant au checkpoint du papier. Présence de code ne prouve pas que le résultat cité a été rerun. |
| Open-set / classe tenue à l'écart | Script et résultats individuels présents. | Fournir la baseline aléatoire et l'agrégation entre classes. Restaurer ou exposer l'accuracy comme critère de sélection pour reproduire les anciens résultats, puis distinguer le protocole macro-F1. |
| Transfert Watkins/BEANS | Runner CPU sur embeddings sauvegardés, manifests, prédictions et bootstrap ajoutés. | Vérifier la provenance des caches et adapter l'extraction audio pour une reproduction complète. Voir `experiments/benchmark/WATKINS.md`. |
| Accord inter-annotateurs | Organisé et vérifié localement dans `rebuttal/inter_annotator_agreement`, absent du dépôt par choix de Pablo. | Garder hors du dépôt pour l'instant ; décider ultérieurement de sa publication. |
| Préentraînement AVES depuis zéro et continu | Évaluation de checkpoints disponible ; pas d'entraînement HuBERT/AVES dans `experiments/pretraining/`. | Intégrer ou référencer les recettes locales : manifests audio, MFCC/k-means, extraction couche HuBERT, apprentissage, BTB3 et conversion/export. |
| Ablation Wav2Vec2 10/50/100 % | Les checkpoints peuvent être évalués ; la création des sous-corpus n'est pas fournie. | Retrouver les manifests et le code de sélection stratifiée année/canal/durée, puis les recettes d'entraînement de chaque budget. |
| Évaluations externes du CNN | Helpers d'installation et inférence présents, mais préparation et scoring délégués à un dossier local DolphinWhistleExtractor. | Fournir le projet externe avec une URL/version fixée ou importer les modules nécessaires et leurs manifests. |
| Pipeline d'annotation | Détection et regroupement de séquences présents. F0 et ARTwarp ne sont pas implémentés dans OpenWhistle. | Documenter et fournir extraction F0/poids/contours, dépendances ARTwarp, paramètres et templates. |
| Figures | Figures statistiques 2–3 et matrice de confusion disponibles. | Les CSV permettent le rendu, mais certains calculs d'origine dépendent d'autres outils. Ajouter la sélection des exemples spectrogrammes de la figure 7B si l'objectif est de régénérer toute la figure. Les illustrations/photos/cartes ne nécessitent pas forcément du code. |

La dérive temporelle et la modélisation du contexte séquentiel sont décrites comme
directions futures dans le rebuttal. Leur absence ne bloque pas la reproduction
des expériences effectivement présentées.

## Code retrouvé sur le PC

### Accord inter-annotateurs — correspondance vérifiée

- `/home/pablo/Documents/rebuttal/generate_agreement_plots.py`
- `/home/pablo/Documents/rebuttal/agreement_pairs_100.csv`
- `/home/pablo/Documents/rebuttal/agreement_summary_100.json`
- `/home/pablo/Documents/rebuttal/annotations-rebuttal-space/`

Le CSV contient 100 paires. Le résumé donne, toutes paires incluses : accord
0,73, kappa 0,65935, AC1 0,70686. Après exclusion des paires contenant
« Uncertain » : 84 paires, accord 0,82143, kappa 0,76, AC1 0,80716.
Cela correspond aux chiffres arrondis du rebuttal. Le script actuel repart
du fichier de 99 paires et ajoute une paire particulière en dur : une version
publique devrait prendre directement le CSV final et rendre cette provenance
explicite, plutôt que modifier les données à chaque exécution.

### Watkins/BEANS — sources retrouvées, incohérence de métrique

Sources :
`/home/pablo/Documents/openwhistle-aves-jz-full/experiments/wise_alpha/benchmarks/`.
Le dossier contient des manifests, des runners et les exports d'évaluation.
`run_beans_local.py` est principalement un sweep WiSE ;
`scripts/run_hf_hubert_linear_probe.py` contient le probe sur CSV. Son protocole
local sélectionne C par accuracy, utilise une grille plus large, un poids de
classes équilibré, les splits du manifest et une normalisation entraînée sur
train. Une copie de ce script ne suffit pas à reproduire une affirmation de
« même protocole OpenWhistle » sans documenter ces différences.

Résultats précis retrouvés :

| JSON local sous `outputs/beans_watkins_classification/` | Accuracy test | Macro-F1 test |
|---|---:|---:|
| `dolphinteam_OpenWhistle_Wav2Vec2.0/metrics.json` | 76,9912 % | 76,0820 % |
| `aves_bio_rerun/metrics.json` | 78,4661 % | 78,1603 % |

Les valeurs annoncées dans le rebuttal, 76,99 et 78,47, correspondent donc
à l'accuracy dans ces rapports, alors que le texte les appelle macro-F1.
À confirmer avec les runs finaux avant correction du manuscrit. Les JSON
retrouvés ne contiennent pas les écarts-types 2,29 et 2,23 : leur calcul
doit également être retrouvé ou fourni.

Mise à jour du 5 octobre : `train_lr_watkins.py` sélectionne C par macro-F1 de
validation sur les embeddings sauvegardés. Les nouveaux résultats sont
75,88 ± 2,71 % (Wav2Vec2, C=3) et 77,58 ± 2,73 % (AVES-bio, C=1).
Les ± sont des écarts-types bootstrap sur 1 000 resamples du test, avec un
univers fixe de 31 classes ; ils ne reproduisent pas les ± historiques.
L'ancien calcul sélectionné par accuracy est reproduit exactement pour AVES.
Pour Wav2Vec2, le nouveau calcul diffère d'un clip correctement prédit même
à C=3. La cause précise reste à établir ; plusieurs environnements installés
donnent des résultats différents. Les versions et les hashes des entrées du
nouveau calcul sont enregistrés. L'extraction audio n'a pas été relancée.

### Préentraînement AVES / BTB3

- `/home/pablo/Documents/OpenWhistle-Aves/TRAIN_OPENWHISTLE_FROM_SCRATCH.md`
- `/home/pablo/Documents/OpenWhistle-Aves/scripts/` et `recipes/fairseq/`
- `/home/pablo/Documents/openwhistle-aves-jz-full/experiments/stage2_960_jz/`
- `/home/pablo/Documents/openwhistle-aves-jz-full/experiments/stage2_320_jz/`
- `/home/pablo/Documents/openwhistle-aves-jz-full/experiments/beyond_baseband_pretraining/`

Ces sources comportent apprentissage Fairseq, pseudo-labels, configurations
et jobs SLURM. Elles nécessitent une adaptation des chemins et une version
explicite des dépendances externes. Ne pas copier tout le dossier, qui contient
aussi des poids et des expériences sans rapport avec le rebuttal.

Les deux checkpoints AVES Stage-1/Stage-2 figurent dans le CSV importé et
sont désormais inclus dans `FROZEN_COLLECTION_MODELS`, avec un loader natif
TorchAudio et des révisions fixées. Ils restent hors de la liste Transformers
utilisée pour le fine-tuning. Le Stage-2-320 correspond aux valeurs enregistrées
61,26 F1 / 76,18 mAP du modèle from-scratch. Le checkpoint
BTB3 100 % correspond aux valeurs 63,53 / 73,93 du modèle continu.

Pour l'ablation Wav2Vec2, le checkpoint étiqueté stride960 100 % dans la
collection donne 60,57 F1 ; le résultat 64,37 provient de
`dolphinteam/OpenWhistle-Wav2Vec2.0`. La recette de table doit préciser ce
choix et ne pas confondre les deux variantes.

La table `rebuttal_pretraining.md` et le preset `--collection rebuttal_pretraining`
font désormais ce choix explicitement. Les valeurs 22,65 / 56,53 / 64,37 du
rebuttal sont retrouvées dans le CSV, sans nouvel entraînement. Un smoke test
CPU du loader AVES avec les vrais poids Stage-2 passe ; le SHA-256 de ces poids
correspond à celui déclaré par le Hub. L'accès Hub vérifié le 5 octobre montre
que les AVES natifs et Wav2Vec2 d'ablation sont privés ; seul le checkpoint
Wav2Vec2 principal est public parmi les six vérifiés.

Le config du Wav2Vec2 principal contient une référence `checkpoint-182000`,
alors que le preprint décrit 400k steps : retrouver le trainer state et la
règle de sélection avant de conclure sur ce décalage. Voir
`experiments/benchmark/PRETRAINING.md` et `docs/provenance/pretraining-checkpoints.json`.

Les scripts `prepare_stage1_fraction.py` et
`prepare_btb3_fraction_data.py` sélectionnent un préfixe de manifests ;
ils ne démontrent pas eux-mêmes la stratification année/canal/durée annoncée
pour l'ablation Wav2Vec2. Le générateur exact de ces sous-corpus n'a pas été
identifié dans les dossiers inspectés.

### CNN externe et extraction F0

- `/home/pablo/Documents/DolphinWhistleExtractor/external_benchmarks/wmmsd/`
- `/home/pablo/Documents/DolphinWhistleExtractor/external_benchmarks/dclde/`
- `/home/pablo/Documents/DolphinWhistleExtractor/scripts/rebuild_classification_f0_dataset.py`
- `/home/pablo/Documents/Dolph2Vec/users/pablo/bioacoustic_F0_estimation/`

Les modules externes contiennent préparation et scoring, ce qui manque au
helper OpenWhistle pris seul. L'extraction F0 utilise des poids CREPE locaux
à identifier et rendre accessibles. Le dernier dossier est effectivement
non suivi dans le dépôt local Dolph2Vec. Aucun pipeline ARTwarp complet avec
templates n'a été identifié dans les sources inspectées.

### Autres modifications Git locales

`Dolph2Vec` comporte des modifications non commités dans son préentraînement
et le dossier F0 non suivi. `SoundPretraining` comporte un launcher modifié
et `run_local_rguerrer.py` non suivi. Ces changements doivent être comparés
à la recette utilisée pour les checkpoints du rebuttal ; leur existence ne
suffit pas à les considérer comme la version finale. `DolphinWhistleExtractor`
est propre dans Git : son code manque à OpenWhistle, mais n'est pas non commité
dans son propre dépôt. `rebuttal` et `OpenWhistle-Aves` n'ont pas de `.git`.

## Corrections de recette identifiées dans OpenWhistle

1. `submit_dolphin.py` remplace sans condition les arguments de chemin modèle,
   préprocesseur et training YAML par ses fichiers intégrés. Il faut respecter
   les overrides pour lancer les ablations.
2. Le launcher par défaut demande 8 × 8 = 64 GPU. La recette préprint en décrit
   32. Les paramètres YAML essentiels sont présents, mais la commande exacte
   doit fixer la topologie et donc le batch effectif.
3. `run_all.sh` ne sélectionne pas explicitement `all` et les plotters ne
   proposent pas tous le choix de configuration/métrique nécessaire.
4. Ajouter les baselines aléatoires et une agrégation des résultats par classe
   tenue à l'écart ; ne pas confondre dispersion entre classes, entre seeds
   et bootstrap.
5. Fixer les revisions de datasets et de modèles, publier les manifests des
   sous-corpus/splits et donner un environnement versionné.

## Ordre de travail proposé

1. Résoudre les métriques et la correspondance checkpoint–ligne de tableau,
   notamment Watkins, AVES scratch et Wav2Vec2 100 %.
2. Compléter l'extraction et la provenance Watkins ; garder l'accord
   inter-annotateurs organisé hors dépôt selon la demande de Pablo.
3. Fournir les recettes AVES, BTB3 et Wav2Vec2 par budget de données.
4. Fournir préparation/scoring CNN externe et les dépendances du pipeline
   d'annotation.
5. Ajouter des commandes et tables de reproduction pour chaque expérience,
   puis effectuer des smoke tests dans les environnements adaptés.

Cet audit distingue le code intégré, les recalculs sur caches et les sources
externes encore à adapter. Il ne constitue pas une validation GPU complète.
