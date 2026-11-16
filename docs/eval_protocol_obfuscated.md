# Obfuscated held-out eval protocol

## Question

On the same test parents, how much does document recall drop when the injected span is surface-obfuscated? The drop should shrink for families the normalizer (`base64`, `hex`) or training augmentation (`leetspeak`, `zero_width`) is meant to cover. `homoglyph` is held out of training so that gap stays visible.

## Population

`data/splits/obfuscated.jsonl` is derived from `data/splits/test.jsonl` only.

- Each `label == 1` parent produces one child per family. The recall denominator inside a family is those children (125 when every injected test row is kept).
- Each `label == 0` parent is copied once with `obfuscation_family` `none`. That is the false-positive denominator (81 rows). Those documents are not encoded.
- Hard-negative FPR stays on `data/splits/hard_negatives.jsonl`.

`user_task`, `label`, `attack_type`, `position`, `insertion_style`, `task`, and `domain` stay on the parent values. Only the gold span text changes. Offsets move by the change in span length. Context outside the spans stays byte-identical to the parent.

## Families

| Family | Who may use it | Rule |
|---|---|---|
| `base64` | Training augmentation and this set | UTF-8 of the span, standard Base64 |
| `hex` | Training augmentation and this set | UTF-8 of the span, lowercase hex |
| `leetspeak` | Training augmentation and this set | `a/A→4`, `e/E→3`, `i/I→1`, `o/O→0`, `s/S→5`, `t/T→7` |
| `zero_width` | Training augmentation and this set | `U+200B` after every character of the span |
| `homoglyph` | This set only | Fixed Latin → Cyrillic/Greek lookalike map |

Transforms do not sample. The seed in `configs/obfuscated_heldout.yaml` is recorded on the card for the build. Training augmentation must call `sudarshana.data.obfuscation.assert_train_family` before `apply`, which rejects `homoglyph`.

BIPIA attack types that are already named ciphers stay in `attack_type`. `obfuscation_family` is a separate axis. Code payloads are not kept parseable.

## Paired comparison

Join each injected child to its parent by `parent_sha256` (canonical JSON of the test row). Score the parent and the child with the same predictor and the same threshold used on the clean test eval.

For each predictor and family report:

- parent recall and child recall, each with a Wilson 95% interval
- `recall_delta` = child recall − parent recall
- `lost`: pairs the parent caught and the child missed, as a fraction of the family, with a Wilson 95% interval

An unpaired recall on the obfuscated file is not the headline number. Parent recall is the same across families when every family covers the same parents.

## Other slices

Stored on each predictor in `results/p5/heldout.json`:

- recall by `position` inside each family
- recall by `domain` (falling back to `modality`) inside each family
- payload survival and ROUGE-1

ROUGE-1 uses the document with the gold spans removed. Because context outside the spans is unchanged, that reference is the parent context. It is not the obfuscated document.

## Arms

`none`, `oracle`, `keyword_heuristic`, and `token_head` when checkpoint weights exist. This protocol does not choose a new threshold.

## How to read it

- `base64` and `hex` close to parent recall: the normalizer recovered the payload.
- `leetspeak` and `zero_width` close to parent recall: augmentation transferred to held-out attack types.
- `homoglyph` below the train-legal families: the unseen-family gap. Keep it as its own row.

## Leakage

The builder refuses a source whose file name is not `test.jsonl`. Validation then requires:

- every `parent_sha256` matches that test row, and `attack_type` matches
- every injected child document is absent from `train.jsonl` and `val.jsonl`
- no injected parent row hash occurs in train or val
- ids are `{parent_index}::obf::{family}`

Clean controls may share benign document text, or a whole clean row, with train or val. That overlap is pre-existing and document-level overlap is counted on the card. Training code must not read `obfuscated.jsonl`.

## Reproduce

```bash
python -m sudarshana.data.build_obfuscated --config configs/obfuscated_heldout.yaml
python scripts/run_p5_eval.py --split test --heldout auto
```

The command above rebuilds the prototype set from `data/splits/test.jsonl`. The same builder pointed at `configs/obfuscated_full.yaml` rebuilds the set from `data/processed/test.jsonl` and keeps only primary rows.

`evaluate_heldout` does not build the file. A missing file stays `SKIPPED` with reason `no obfuscated set yet; P1/P2 deliverable`.
