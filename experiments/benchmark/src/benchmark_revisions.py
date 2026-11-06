"""Hub revisions used by the published frozen-probe benchmark.

Local embedding cache identities include the revisions recorded here.
Native AVES revisions and weight hashes live in aves_models.py.
"""

MODEL_REVISIONS = {
    "dolphinteam/OpenWhistle-Wav2Vec2.0": "b5f156048fc4da1d8d8b7345b61ebef1257b365c",
    "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz": "9f5de05d665e69118a4f9ba921beaf54f6dd58d9",
    "davidrrobinson/biolingual": "808bb50859ce7d0c0fcc2b233676c7ba9319107e",
}

DATASET_REVISIONS = {
    # The published scores were computed at 2b7806e3c5913c459c40b824ab7f944d494b3993,
    # where the unbalanced config was named "all". Its parquet files and those of
    # the balanced config are byte-identical (same SHA-256) at this revision.
    "dolphinteam/OpenWhistle-Classification-Finetuning": "1026daf811f1f134d98a977e0cdd1e1129692497",
    "dolphinteam/OpenWhistle-Detection-Finetuning": "c2e89807b51142e106b76d486f4cf620ec63c1fb",
}
