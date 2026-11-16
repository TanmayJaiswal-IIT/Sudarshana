# Obfuscated held-out card

Surface-obfuscated copy of the test split. Evaluation reads this file.
Training must not. Reserved families must not be used as training augmentation.

- Source: `data/processed/test.jsonl`
- Source sha256: `37ec98db07f02130058ee3c6e488118c0da048da9551cfacf461262c32d76bfe`
- Seed: 20261115 (recorded; the current transforms do not sample)
- Git revision: `cb909104feca756e4bcfa1299246f76a495221de`
- Rows: 210839
- Injected parents: 41960
- Clean controls: 1039
- Reserved families: `homoglyph`
- Clean controls whose document text also appears in train: 277
- Clean controls whose document text also appears in val: 129
- Primary rows only: True

## Counts by family

| Family | Rows | Unchanged payloads |
|--------|------|--------------------|
| none | 1039 | — |
| base64 | 41960 | 0 |
| hex | 41960 | 0 |
| leetspeak | 41960 | 0 |
| zero_width | 41960 | 0 |
| homoglyph | 41960 | 0 |

Protocol: `docs/eval_protocol_obfuscated.md`
