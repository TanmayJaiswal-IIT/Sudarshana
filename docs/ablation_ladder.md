# FPR calibration and ablation ladder

## Threshold

The detection threshold is the lowest validation score whose false-positive rate is at most **0.05**. The score is the lexical log-odds, before the sigmoid. The sigmoid maps large positive log-odds to exactly 1, so a probability threshold cannot separate those documents. A score greater than or equal to the threshold is a detection. The fit uses train rows only. Validation rows with no `id` are included. Rows whose `id` contains `::` are not used to fit or to choose the threshold.

Test, `hard_negatives.jsonl`, and `obfuscated.jsonl` are scored once the threshold is frozen. Hard-negative false-positive rate is reported and is not the cap. With 23 validation negatives, 0.05 means at most one false positive. `tight` is true when the next-lower candidate exceeds the cap.

The Day 3 point, the highest threshold that still reaches 95% validation recall, is stored as `recall_reference` on each arm. The frozen number used for test and the held-out set is the false-positive cap, not that recall point.

The redaction cutoff stays the value in `configs/prototype_config.yaml` (`redaction.calibrated_threshold`, 0.35). It is a token-span knob. This procedure does not change it.

## Ladder

Each rung is the previous rung plus one change. A rung is `OK` only when its component exists and every earlier rung is `OK`. Otherwise it is `NOT_RUN` and it has no test recall. A missing rung is not filled with another model's scores.

| Rung | Component |
|---|---|
| 1 Baseline | Truncated lexical model, document only, 512 tokens |
| 2 + Normalizer | `sudarshana.preprocessing.normalizer.normalize` |
| 3 + Chunking | `sudarshana.preprocessing.chunker.chunk_document` |
| 4 + Pair input | `sudarshana.preprocessing.pair_chunks.score_pair_chunks` |
| 5 + Token head and redaction | A checkpoint weight file under `checkpoints/` |
| 6 + Obfuscation augmentation | `sudarshana.training.augment.augment_train` |

`available_arms` calibrates the document lexical model and the pair lexical model independently at the same 0.05 cap. The pair arm is not rung 4. Rung 4 is pair input on top of the normalizer and the chunker.

`frozen_rung` is the last `OK` rung. Later code registers a stacked scorer by adding the module named above. The cap does not change.

## Reproduce

```bash
python -m sudarshana.evaluation.ablation --config configs/ablation_ladder.yaml
```

Writes `results/calibration_day6.json` and `results/ablation_ladder.json`. That config reads the prototype files in `data/splits` (test n=206).

The authoritative split is `data/processed`. Rebuild the obfuscated set from its test file, then calibrate and coordinate the ladder there:

```bash
python -m sudarshana.data.build_obfuscated --config configs/obfuscated_full.yaml
python -m sudarshana.evaluation.ablation --config configs/ablation_full.yaml
```

`primary_only` drops rows whose id contains `::` (counterfactual and mismatched siblings). Held-out hard negatives are the test rows with source `hard_negative` when `hard_negatives.jsonl` is absent. Writes `results/calibration_full.json` and `results/ablation_ladder_full.json`.
