# P1 full-split results

Day 4 span QA already locked `data/processed`. This note is the Day 5 obfuscated held-out set and the Day 6 false-positive calibration on that same split. The prototype files in `data/splits` and `results/calibration_day6.json` are unchanged. P5's n=206 table is a different corpus.

Score is the truncated lexical log-odds, before the sigmoid. The model sees the first 512 whitespace tokens. Fit uses primary train rows only (ids with no `::`). The threshold is the lowest validation score whose false-positive rate is at most 0.05. `tight` means the next-lower candidate exceeds that cap. The 95% recall threshold is recorded and is not applied to test. Held-out hard-negative FPR is reported and is not the cap. The redaction cutoff stays 0.35.

Git revision of the run: `cb909104feca756e4bcfa1299246f76a495221de`.

## Splits

| Split | Rows in file | Rows used |
|-------|--------------|-----------|
| Train | 244107 | 179820 primary |
| Validation | 60639 | 44729 primary (15910 ids with `::` left out of the threshold) |
| Test | 62859 | 42999 primary |
| Hard negatives | no sidecar | 400 primary test rows with source `hard_negative` |
| Obfuscated | 210839 | 1039 clean controls + 41960 parents × 5 families |

Primary test is 1039 label-0 rows and 41960 label-1 rows. That is the n=42999 the open item was tracking.

## Obfuscated held-out

Built with `configs/obfuscated_full.yaml` (`primary_only: true`) from `data/processed/test.jsonl`. Card: `reports/obfuscated_full_card.md`. The JSONL is `data/processed/obfuscated.jsonl`.

Every injected payload changed under base64, hex, leetspeak, zero_width, and homoglyph (0 unchanged). `homoglyph` stays reserved from training augmentation. Leakage check: no injected child document and no injected parent row appears in train or val. Clean controls whose document text also appears elsewhere: 277 in train, 129 in val. That overlap is counted and not rejected. `missing_parents` at scoring time is 0.

## Document arm (frozen rung)

Threshold **117.01904070715975**, tight, validation FPR 0.0495 (16/323).

| Set | Recall | FPR |
|-----|--------|-----|
| Validation | 0.0229 (1017/44406) | 0.0495 (16/323) |
| Test | 0.0000 (0/41960) | 0.0000 (0/1039) |
| Hard negatives | — (0 positives) | 0.0000 (0/400) |

The 95% validation-recall reference sits at threshold 29.706948820245575 with FPR 0.5882 (190/323) and recall 0.95 (42187/44406). The 5% cap is the operating point that was frozen. At that cap the primary test set has no detections.

Paired obfuscated recall, parent and child, is 0.0000 on all five families (n=41960 each), so recall delta is 0.0000. Clean obfuscated controls: FPR 0.0000 (0/1039).

## Pair arm

Same cap, user task then document, truncated once. This arm is in `available_arms`. It is not rung 4, because the normalizer and chunker have not run.

Threshold **95.46412379403974**, tight, validation FPR 0.0495 (16/323), validation recall 0.0178 (792/44406). Test recall 0.0000 (0/41960), test FPR 0.0000 (0/1039), hard-negative FPR 0.0000 (0/400). The 95% recall reference is threshold 5.19402511177734 at FPR 0.6068 (196/323). Obfuscated recall delta is 0.0000 on every family.

## Ablation ladder

Frozen rung: `1_baseline`.

| Rung | Status | Reason |
|------|--------|--------|
| 1 Baseline, truncated document | OK | numbers above |
| 2 Normalizer | NOT_RUN | `sudarshana.preprocessing.normalizer` is absent |
| 3 Chunking | NOT_RUN | no chunker module |
| 4 Pair input | NOT_RUN | waits on rungs 2 and 3; lexical pair arm is above |
| 5 Token head | NOT_RUN | no checkpoint weights |
| 6 Obfuscation augmentation | NOT_RUN | `sudarshana.training.augment` is absent |

## Prototype record

`results/calibration_day6.json` is the n=206 run: document threshold 71.51743049192336, validation recall 0.0267, validation FPR 0.0435, test recall 0.0000 on 206 rows, hard negatives n=15. Those files were not overwritten.

## Reproduce

```bash
python -m sudarshana.data.build_obfuscated --config configs/obfuscated_full.yaml
python -m sudarshana.evaluation.ablation --config configs/ablation_full.yaml
```

Writes `results/calibration_full.json` and `results/ablation_ladder_full.json`.
