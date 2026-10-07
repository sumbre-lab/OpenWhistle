# Vérification du préentraînement sur Jean Zay

Vérification du 7 octobre 2026, par SSH en lecture seule. Aucun job lancé,
aucune modification sur le cluster et aucun upload Hugging Face.

## Conclusion

Le dossier `experiments/pretraining/` correspond au pipeline Transformers du
modèle principal. Il ne contient pas le pipeline Fairseq qui a produit les
checkpoints Wav2Vec2 à 10 % et 50 % utilisés dans le rebuttal.

| Modèle Hugging Face (`dolphinteam/`) | Source des poids sur Jean Zay | Vérification |
|---|---|---|
| `OpenWhistle-Wav2Vec2.0` | `/lustre/fswork/projects/rech/ioc/commun/model-dolph2vec_type-base_data-DolphinChat_version-v0/model.safetensors` | SHA-256 identique au Hub et aux poids locaux |
| `wav2vec2-44k-stride960-10pct-40k` | `Wav2Vec2.0/outputs/hf_exports/wav2vec2-44k-stride960-10pct-40k/model.safetensors` sous le même dossier `commun` | SHA-256 identique au Hub et aux poids locaux |
| `wav2vec2-44k-stride960-50pct-200k` | `Wav2Vec2.0/outputs/hf_exports/wav2vec2-44k-stride960-50pct-200k/model.safetensors` sous le même dossier `commun` | SHA-256 identique au Hub et aux poids locaux |

Les hashes Hub sont ceux vérifiés le 5 octobre 2026. Les hashes des poids
Jean Zay ont été calculés le 7 octobre. Les preuves et paramètres sélectionnés
sont dans [jeanzay-pretraining-verification.json](jeanzay-pretraining-verification.json).

## Modèle principal : poids et paramètres effectifs

Le `trainer_state.json` accompagnant les poids correspondants indique
**191 000 étapes réalisées**, pour **400 000 étapes prévues**. Le dernier log
enregistré est lui aussi à 191 000. Le `training_args.bin` confirme :

- 32 processus distribués ; batch par GPU 4 ; accumulation 2 ; batch effectif 256 ;
- learning rate 0,0005 ; warmup 32 000 ; AdamW ; scheduler linéaire ;
- FP16 ; weight decay 0,01 ; seed 42.

Les 400k étapes du YAML sont donc un objectif, pas le nombre d'étapes du
checkpoint effectivement publié. La référence `checkpoint-182000` dans le
config n'est pas un compteur fiable : le poids identique retrouvé sur le
cluster est accompagné d'un état à 191k. Il faut corriger le manuscrit et
documenter l'historique de reprise/export avant d'affirmer 400k réalisées.

L'architecture principale et les paramètres de masquage inspectés concordent
avec `config_dolphin.json` : 12 couches, dimension 768, 8 têtes, stride 960,
masquage temporel de probabilité 0,65 et longueur 10.

## Comparaison du code actuel

Sur Jean Zay, `/lustre/fswork/projects/rech/ioc/commun/OpenWhistle` est sur
`pablo/main`, commit `e432f97`, avec un launcher et un README modifiés localement.
Sur les 19 fichiers du package de préentraînement comparés à `camera-ready/pablo`,
18 sont identiques octet pour octet, dont le modèle, le trainer, la boucle
d'entraînement et les trois configurations. Le seul fichier différent est
`submit_dolphin.py` : défauts Jean Zay `gpu_p2` / `ioc@v100`, ancien namespace
du dataset, différences de présentation. Le code local laisse les paramètres
de cluster configurables et utilise le namespace `dolphinteam`.

Cette égalité prouve la concordance du code **actuellement présent** sur le
cluster. Elle ne prouve pas une identité avec la révision historique du job
ayant produit les poids. Le dépôt `Dolph2Vec/pretraining` est également présent
et diffère du code actuel : noms de classes, logging, chargement des données
et certains paramètres par défaut. Ne pas le recopier comme recette exacte
sans retrouver la révision ou les sources archivées du job original.

## Ablations : autre pipeline à intégrer

Les poids 10 % et 50 % correspondent aux exports de
`/lustre/fswork/projects/rech/ioc/commun/Wav2Vec2.0`. Ses sources utiles sont :

- `configs/pretraining/wav2vec2_base_44k_stride960.yaml` ;
- `scripts/train_wav2vec2_44k_stride960.sh` ;
- `scripts/slurm/submitit_train_wav2vec2_44k_stride960_jz.py` ;
- `scripts/prepare_openwhistle_fairseq.py` ;
- `scripts/export_fairseq_wav2vec2_pretraining_to_hf.py`.

Ce pipeline utilise Fairseq/Hydra puis convertit les poids en Transformers.
Il est absent du dossier `experiments/pretraining/` actuel. Les sources ciblées
ont été conservées temporairement pour comparaison, sans importer de poids.
La prochaine intégration doit comprendre ce pipeline, ses dépendances et les
manifests effectivement utilisés. Leurs règles de sélection, la stratification
et le checkpoint Fairseq exact choisi restent à auditer.
