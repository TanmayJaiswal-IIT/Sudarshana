# P5 Evaluation Report — Sudarshana Indirect Prompt-Injection Detection

## 1. Setup & Metric Definitions

**Dataset**: `data/splits/test.jsonl` (test split, seed=42)
**Schema**: v1.1 — fields: `user_task`, `document`, `label`, `injection_span`, `attack_type`, `position`
**injection_span**: list of `[start, end]` character-offset pairs into `document`

### Predictors evaluated

| Name | Description |
|------|-------------|
| none | No defence — always predicts 0 (no injection). Reports baseline FPR=0, recall=0. |
| oracle | Gold labels (upper-bound ceiling). |
| keyword_heuristic | P5 lexical keyword scorer (threshold=0.5, document-only). NOT the team baseline. |
| token_head | N/A (no weights — checkpoints contain only config+tokenizer files) |

### ⚠️ Proxy Metric Warnings

**PSR ("ASR proxy — payload survival rate")** is NOT real Attack Success Rate.
Real ASR requires an LLM to execute the injected instruction and confirm behavioural change.
PSR only measures whether the injection *text* survives in the post-redaction document.

**ROUGE-1 F1 ("ROUGE-1 utility proxy")** is NOT real downstream task utility.
Real utility requires running the downstream agent/LLM on the redacted document and
comparing task completion. ROUGE-1 is a surface-text overlap proxy only.

---

## 2. Team Baseline (lexical model, `baseline_day2.json` — teammate result, not recomputed)

> **Ran on full dataset** (train=179 820 / val=44 729 / test=42 999). Committed sample splits
> (test=206, hard_neg=15) are **not comparable**.

| Split | N | Recall | FPR | Precision | F1 |
|-------|---|--------|-----|-----------|-----|
| val | 44729 | 0.9982 | 0.6780 | 0.9951 | 0.9967 |
| test | 42999 | 0.9964 | 0.4456 | 0.9890 | 0.9927 |

**Recall by attack type (val split)**

| Attack Type | Support | Recall |
|-------------|---------|--------|
| Cryptocurrency Mining | 750 | 0.9800 |
| Device and Driver Enumeration | 750 | 1.0000 |
| Malware Distribution | 14250 | 1.0000 |
| Programming Help | 14250 | 1.0000 |
| Space Removal & Grouping | 14250 | 1.0000 |
| injection_task_0 | 40 | 0.0250 |
| prompt_hijack | 116 | 0.7931 |

### P5 keyword_heuristic re-evaluation on `test` split (N=125)

- Recall: 0.0400 [0.000, 1.000] (n=0)
- FPR (test): 0.0123 [0.000, 1.000] (n=0)
- FPR (hard-neg N=15, indicative only): 0.0000

---

## 3. Recall by Injection Position

![Recall by Injection Position](reports/figures/recall_by_position.png)
*Recall by Injection Position*

| Predictor | Position | Support | Recall |
|-----------|----------|---------|--------|
| none | end | 53 | 0.0000 |
| none | middle | 30 | 0.0000 |
| none | start | 42 | 0.0000 |
| oracle | end | 53 | 1.0000 |
| oracle | middle | 30 | 1.0000 |
| oracle | start | 42 | 1.0000 |
| keyword_heuristic | end | 53 | 0.0377 |
| keyword_heuristic | middle | 30 | 0.0333 |
| keyword_heuristic | start | 42 | 0.0476 |


---

## 4. Recall by Attack Category

![Recall by Attack Type](reports/figures/recall_by_attack_type.png)
*Recall by Attack Type*

| Predictor | Attack Type | Support | Recall |
|-----------|-------------|---------|--------|
| none | Base Encoding | 5 | 0.0000 |
| none | Blocking Internet Connection | 5 | 0.0000 |
| none | Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 |
| none | Business Intelligence | 5 | 0.0000 |
| none | Compromising Computers | 5 | 0.0000 |
| none | Conversational Agent | 5 | 0.0000 |
| none | Corrupting an Operating System | 5 | 0.0000 |
| none | Data Eavesdropping | 5 | 0.0000 |
| none | Emoji Substitution | 5 | 0.0000 |
| none | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.0000 |
| none | Entertainment | 5 | 0.0000 |
| none | Information Dissemination | 5 | 0.0000 |
| none | Introduce System Fingerprinting | 5 | 0.0000 |
| none | Keylogging | 5 | 0.0000 |
| none | Language Translation (Test) | 5 | 0.0000 |
| none | Marketing & Advertising | 5 | 0.0000 |
| none | Misinformation & Propaganda | 5 | 0.0000 |
| none | Research Assistance | 5 | 0.0000 |
| none | Reverse Text | 5 | 0.0000 |
| none | Scams & Fraud | 5 | 0.0000 |
| none | Screen Scraping | 5 | 0.0000 |
| none | Sentiment Analysis | 5 | 0.0000 |
| none | Substitution Ciphers | 5 | 0.0000 |
| none | Task Automation | 5 | 0.0000 |
| none | Traffic Analysis | 5 | 0.0000 |
| oracle | Base Encoding | 5 | 1.0000 |
| oracle | Blocking Internet Connection | 5 | 1.0000 |
| oracle | Bringing Down Hosts and Servers (Denial of Service) | 5 | 1.0000 |
| oracle | Business Intelligence | 5 | 1.0000 |
| oracle | Compromising Computers | 5 | 1.0000 |
| oracle | Conversational Agent | 5 | 1.0000 |
| oracle | Corrupting an Operating System | 5 | 1.0000 |
| oracle | Data Eavesdropping | 5 | 1.0000 |
| oracle | Emoji Substitution | 5 | 1.0000 |
| oracle | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 1.0000 |
| oracle | Entertainment | 5 | 1.0000 |
| oracle | Information Dissemination | 5 | 1.0000 |
| oracle | Introduce System Fingerprinting | 5 | 1.0000 |
| oracle | Keylogging | 5 | 1.0000 |
| oracle | Language Translation (Test) | 5 | 1.0000 |
| oracle | Marketing & Advertising | 5 | 1.0000 |
| oracle | Misinformation & Propaganda | 5 | 1.0000 |
| oracle | Research Assistance | 5 | 1.0000 |
| oracle | Reverse Text | 5 | 1.0000 |
| oracle | Scams & Fraud | 5 | 1.0000 |
| oracle | Screen Scraping | 5 | 1.0000 |
| oracle | Sentiment Analysis | 5 | 1.0000 |
| oracle | Substitution Ciphers | 5 | 1.0000 |
| oracle | Task Automation | 5 | 1.0000 |
| oracle | Traffic Analysis | 5 | 1.0000 |
| keyword_heuristic | Base Encoding | 5 | 0.0000 |
| keyword_heuristic | Blocking Internet Connection | 5 | 0.0000 |
| keyword_heuristic | Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 |
| keyword_heuristic | Business Intelligence | 5 | 0.0000 |
| keyword_heuristic | Compromising Computers | 5 | 0.0000 |
| keyword_heuristic | Conversational Agent | 5 | 0.0000 |
| keyword_heuristic | Corrupting an Operating System | 5 | 0.4000 |
| keyword_heuristic | Data Eavesdropping | 5 | 0.0000 |
| keyword_heuristic | Emoji Substitution | 5 | 0.0000 |
| keyword_heuristic | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.2000 |
| keyword_heuristic | Entertainment | 5 | 0.0000 |
| keyword_heuristic | Information Dissemination | 5 | 0.0000 |
| keyword_heuristic | Introduce System Fingerprinting | 5 | 0.2000 |
| keyword_heuristic | Keylogging | 5 | 0.0000 |
| keyword_heuristic | Language Translation (Test) | 5 | 0.0000 |
| keyword_heuristic | Marketing & Advertising | 5 | 0.0000 |
| keyword_heuristic | Misinformation & Propaganda | 5 | 0.0000 |
| keyword_heuristic | Research Assistance | 5 | 0.0000 |
| keyword_heuristic | Reverse Text | 5 | 0.0000 |
| keyword_heuristic | Scams & Fraud | 5 | 0.0000 |
| keyword_heuristic | Screen Scraping | 5 | 0.2000 |
| keyword_heuristic | Sentiment Analysis | 5 | 0.0000 |
| keyword_heuristic | Substitution Ciphers | 5 | 0.0000 |
| keyword_heuristic | Task Automation | 5 | 0.0000 |
| keyword_heuristic | Traffic Analysis | 5 | 0.0000 |


---

## 5. FPR on Hard Negatives

| Predictor | FPR (test negatives) | FPR (hard negatives) |
|-----------|----------------------|----------------------|
| none | 0.0000 | 0.0000 |
| oracle | 0.0000 | 0.0000 |
| keyword_heuristic | 0.0123 | 0.0000 |


---

## 6. Downstream: PSR and ROUGE-1 (none vs oracle vs predicted)

![PSR and ROUGE-1 by Predictor](reports/figures/psr_rouge_by_condition.png)
*PSR and ROUGE-1 by Predictor*

| Predictor | PSR (↓ better for defence) | ROUGE-1 F1 (↑ better) |
|-----------|---------------------------|----------------------|
| none | 1.0000 | 0.8962 |
| oracle | 0.0000 | 0.8996 |
| keyword_heuristic | 1.0000 | 0.8967 |


**PSR interpretation**: Lower PSR = defence removed more attack payload.
**ROUGE-1 interpretation**: Higher F1 = more of the clean content preserved.

---

## 7. Latency

![Pipeline Latency p50/p95](reports/figures/latency_p50_p95.png)
*Pipeline Latency p50/p95*

Benchmark: ~1783 estimated tokens (12674 chars). 20 runs, 3 warmups.

| Stage | p50 (ms) | p95 (ms) | Mean (ms) |
|-------|----------|----------|-----------|
| sentence_segmentation | 17.14 | 33.65 | 19.87 |
| redact | 0.06 | 0.10 | 0.07 |
| tokenizer_encode | 3.49 | 4.79 | 3.69 |
| model_forward | N/A | N/A | N/A (no weights — checkpoints/ contain only config+tokenizer files) |


---

## 8. Obfuscated Held-Out Status

Evaluated on `data/splits/obfuscated.jsonl` (706 records).

Child recall is the obfuscated payload. Parent recall is the same predictor on the clean test parent. Delta is child recall minus parent recall. Lost is the fraction of pairs the parent caught and the child missed, with a Wilson 95% interval. Position and domain slices are in this JSON under each predictor.

| Predictor | Family | N | Child recall | Parent recall | Delta | Lost [Wilson] |
|-----------|--------|---|--------------|---------------|-------|---------------|
| none | base64 | 125 | 0.0 | 0.0 | 0.0 | 0.0 [0.0, 0.0298] |
| none | hex | 125 | 0.0 | 0.0 | 0.0 | 0.0 [0.0, 0.0298] |
| none | leetspeak | 125 | 0.0 | 0.0 | 0.0 | 0.0 [0.0, 0.0298] |
| none | zero_width | 125 | 0.0 | 0.0 | 0.0 | 0.0 [0.0, 0.0298] |
| none | homoglyph | 125 | 0.0 | 0.0 | 0.0 | 0.0 [0.0, 0.0298] |
| oracle | base64 | 125 | 1.0 | 1.0 | 0.0 | 0.0 [0.0, 0.0298] |
| oracle | hex | 125 | 1.0 | 1.0 | 0.0 | 0.0 [0.0, 0.0298] |
| oracle | leetspeak | 125 | 1.0 | 1.0 | 0.0 | 0.0 [0.0, 0.0298] |
| oracle | zero_width | 125 | 1.0 | 1.0 | 0.0 | 0.0 [0.0, 0.0298] |
| oracle | homoglyph | 125 | 1.0 | 1.0 | 0.0 | 0.0 [0.0, 0.0298] |
| keyword_heuristic | base64 | 125 | 0.008 | 0.04 | -0.032 | 0.032 [0.0125, 0.0794] |
| keyword_heuristic | hex | 125 | 0.008 | 0.04 | -0.032 | 0.032 [0.0125, 0.0794] |
| keyword_heuristic | leetspeak | 125 | 0.008 | 0.04 | -0.032 | 0.032 [0.0125, 0.0794] |
| keyword_heuristic | zero_width | 125 | 0.008 | 0.04 | -0.032 | 0.032 [0.0125, 0.0794] |
| keyword_heuristic | homoglyph | 125 | 0.008 | 0.04 | -0.032 | 0.032 [0.0125, 0.0794] |

---

## 8a. FPR calibration and ablation ladder

Target false-positive rate 0.05 on `val`. Frozen rung: `1_baseline`.
Redaction cutoff remains 0.35 and is not this detection threshold.

| Rung | Status | Threshold | Val FPR | Val recall | Test recall | Test FPR |
|------|--------|-----------|---------|------------|-------------|----------|
| Baseline — truncated, single string | OK | 71.51743049192336 | 0.0435 | 0.0267 | 0.0 | 0.0 |
| + Normalizer | NOT_RUN | — | — | — | — | — |
| + Chunking with max pooling | NOT_RUN | — | — | — | — | — |
| + Pair input | NOT_RUN | — | — | — | — | — |
| + Token head and redaction | NOT_RUN | — | — | — | — | — |
| + Obfuscation augmentation | NOT_RUN | — | — | — | — | — |

Reasons for rungs that did not run:

- `2_normalizer`: no normalizer module
- `3_chunking`: no chunker module
- `4_pair`: incremental pair rung waits until the normalizer and chunker run; the lexical pair arm is in available_arms
- `5_token_head`: no token-head checkpoint weights
- `6_augmentation`: no training augmenter

Available arms use the same cap on their own. They are not incremental ladder gains.

| Arm | Threshold | Val FPR | Val recall | Test recall | Test FPR | Hard-neg FPR |
|-----|-----------|---------|------------|-------------|----------|--------------|
| document | 71.51743049192336 | 0.0435 | 0.0267 | 0.0 | 0.0 | 0.0 |
| pair | 72.73086911512883 | 0.0435 | 0.0267 | 0.0 | 0.0 | 0.0 |

| Arm | Family | N | Parent recall | Child recall | Delta |
|-----|--------|---|---------------|--------------|-------|
| document | base64 | 125 | 0.0 | 0.0 | 0.0 |
| document | hex | 125 | 0.0 | 0.0 | 0.0 |
| document | leetspeak | 125 | 0.0 | 0.0 | 0.0 |
| document | zero_width | 125 | 0.0 | 0.0 | 0.0 |
| document | homoglyph | 125 | 0.0 | 0.0 | 0.0 |
| pair | base64 | 125 | 0.0 | 0.0 | 0.0 |
| pair | hex | 125 | 0.0 | 0.0 | 0.0 |
| pair | leetspeak | 125 | 0.0 | 0.0 | 0.0 |
| pair | zero_width | 125 | 0.0 | 0.0 | 0.0 |
| pair | homoglyph | 125 | 0.0 | 0.0 | 0.0 |

---

## 9. ⚠️ SIMULATED Redaction Sensitivity Analysis

> All results in this section are **SIMULATED**. Gold spans are perturbed to model imperfect
> redaction (partial span coverage, span drop). Real results require trained span predictions.

**Label**: SIMULATED redaction sensitivity — gold spans perturbed; not real model predictions

### 9a. PSR and ROUGE-1 Grid (keep_fraction f × drop_rate m)

| keep_fraction | drop_rate | PSR [95% CI] (n=injected) | ROUGE-1 F1 |
|---------------|-----------|---------------------------|------------|
| 1.0 | 0.0 | 0.0000 [0.000, 0.030] (n=125) | 0.8996 |
| 1.0 | 0.1 | 0.0000 [0.000, 0.030] (n=125) | 0.8996 |
| 1.0 | 0.25 | 0.0000 [0.000, 0.030] (n=125) | 0.8996 |
| 1.0 | 0.5 | 0.0000 [0.000, 0.030] (n=125) | 0.8996 |
| 0.9 | 0.0 | 0.0000 [0.000, 0.030] (n=125) | 0.9097 |
| 0.9 | 0.1 | 0.0000 [0.000, 0.030] (n=125) | 0.9097 |
| 0.9 | 0.25 | 0.0000 [0.000, 0.030] (n=125) | 0.9097 |
| 0.9 | 0.5 | 0.0000 [0.000, 0.030] (n=125) | 0.9097 |
| 0.75 | 0.0 | 0.0320 [0.013, 0.079] (n=125) | 0.9137 |
| 0.75 | 0.1 | 0.0320 [0.013, 0.079] (n=125) | 0.9137 |
| 0.75 | 0.25 | 0.0320 [0.013, 0.079] (n=125) | 0.9137 |
| 0.75 | 0.5 | 0.0320 [0.013, 0.079] (n=125) | 0.9137 |
| 0.5 | 0.0 | 0.3200 [0.245, 0.406] (n=125) | 0.9402 |
| 0.5 | 0.1 | 0.3200 [0.245, 0.406] (n=125) | 0.9402 |
| 0.5 | 0.25 | 0.3200 [0.245, 0.406] (n=125) | 0.9402 |
| 0.5 | 0.5 | 0.3200 [0.245, 0.406] (n=125) | 0.9402 |

### 9b. Oracle PSR and ROUGE-1 by Attack Type (f=1.0, m=0.0)

![Oracle PSR by Attack Type](reports/figures/psr_by_attack_type.png)
*Oracle PSR by Attack Type*

| Attack Type | N | PSR [95% CI] | ROUGE-1 F1 |
|-------------|---|--------------|------------|
| Base Encoding | 5 | 0.0000 [0.000, 0.434] | 0.9787 |
| Blocking Internet Connection | 5 | 0.0000 [0.000, 0.434] | 0.7917 |
| Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 [0.000, 0.434] | 0.7884 |
| Business Intelligence | 5 | 0.0000 [0.000, 0.434] | 0.9490 |
| Compromising Computers | 5 | 0.0000 [0.000, 0.434] | 0.7775 |
| Conversational Agent | 5 | 0.0000 [0.000, 0.434] | 0.9788 |
| Corrupting an Operating System | 5 | 0.0000 [0.000, 0.434] | 0.9909 |
| Data Eavesdropping | 5 | 0.0000 [0.000, 0.434] | 0.7978 |
| Emoji Substitution | 5 | 0.0000 [0.000, 0.434] | 0.9705 |
| Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.0000 [0.000, 0.434] | 0.7979 |
| Entertainment | 5 | 0.0000 [0.000, 0.434] | 0.9933 |
| Information Dissemination | 5 | 0.0000 [0.000, 0.434] | 0.9309 |
| Introduce System Fingerprinting | 5 | 0.0000 [0.000, 0.434] | 1.0000 |
| Keylogging | 5 | 0.0000 [0.000, 0.434] | 0.6000 |
| Language Translation (Test) | 5 | 0.0000 [0.000, 0.434] | 0.9794 |
| Marketing & Advertising | 5 | 0.0000 [0.000, 0.434] | 0.9534 |
| Misinformation & Propaganda | 5 | 0.0000 [0.000, 0.434] | 0.9854 |
| Research Assistance | 5 | 0.0000 [0.000, 0.434] | 0.9816 |
| Reverse Text | 5 | 0.0000 [0.000, 0.434] | 0.9696 |
| Scams & Fraud | 5 | 0.0000 [0.000, 0.434] | 0.9723 |
| Screen Scraping | 5 | 0.0000 [0.000, 0.434] | 0.7818 |
| Sentiment Analysis | 5 | 0.0000 [0.000, 0.434] | 0.9619 |
| Substitution Ciphers | 5 | 0.0000 [0.000, 0.434] | 0.9885 |
| Task Automation | 5 | 0.0000 [0.000, 0.434] | 0.7825 |
| Traffic Analysis | 5 | 0.0000 [0.000, 0.434] | 0.7879 |

### 9c. Oracle PSR and ROUGE-1 by Position (f=1.0, m=0.0)

| Position | N | PSR [95% CI] | ROUGE-1 F1 |
|----------|---|--------------|------------|
| end | 53 | 0.0000 [0.000, 0.068] | 0.9169 |
| middle | 30 | 0.0000 [0.000, 0.114] | 0.9081 |
| start | 42 | 0.0000 [0.000, 0.084] | 0.8716 |

![Redaction Sensitivity Grid](reports/figures/redaction_sensitivity.png)
*Redaction Sensitivity Grid*

---

## 10. Limitations

1. **Real split sizes differ from team baseline** — P5 evaluates on committed sample splits (test=206, hard_neg=15); team baseline ran on full dataset (test=42 999). Numbers are **not comparable**.
2. **Hard-neg N=15** — only 15 hard-negative records; FPR estimates are indicative only (wide Wilson CIs).
3. **ASR proxy** — PSR measures payload-text survival; real ASR requires LLM execution and behavioural verification.
4. **ROUGE-1 utility proxy** — surface-text overlap; actual task utility requires downstream agent evaluation.
5. **token_head: N/A** — no trained weights; all token-head metrics are N/A.
6. **Obfuscated held-out** — surface-obfuscated test payloads are evaluated in section 8. `homoglyph` is reserved from training augmentation.
7. **No pair-input evaluation** — P3's (user_task, document) pair collator not integrated; all predictors use document-only input.
8. **No LLM ASR** — `LLMBackend` protocol stubbed; `compute_llm_asr` returns N/A.
9. **Sensitivity is SIMULATED** — gold span perturbations, not real predicted spans from a trained model.

---

## 10a. Ablation Ladder (team-reported, P3, n=206, seed=42)

> Source: `reports/ablation_ladder_tables.json` (owner P3, please review)
> Provenance: team-reported (P3-measured). n=206 for all steps.

| Step | Variant | Recall | Clean FPR (N=75) | HN FPR @ 90% Rec | Span F1 | Latency (ms) |
|------|---------|--------|-----------------|-----------------|---------|-------------|
| step1_baseline | step1_baseline | 0.6080 | 0.2667 | 0.0000 | n/a | 43.56 |
| step2_normalizer | step2_normalizer | 0.3520 | 0.1467 | 0.0000 | n/a | 46.12 |
| step3_chunking | step3_chunking | 0.2800 | 0.2133 | 1.0000 | n/a | 152.40 |
| step4_pair_input | step4_pair_input | 0.3520 | 0.2267 | 1.0000 | n/a | 158.85 |
| step5_token_head | step5_token_head | 0.4880 | 0.0667 | 0.8333 | 34.42% | 164.20 |
| step6_augmented | step6_augmented | 0.6160 | 0.1733 | 0.8333 | 34.42% | 166.61 |

---

## 10b. Trained-Model Recall by Position / Attack Type (recomputed, P5)

> Source: `reports/audit/raw_test_predictions.jsonl` (n=206)
> Provenance: recomputed (P5, from raw_test_predictions.jsonl, is_blocked as predicted label)

**Overall** (n=206, provenance: recomputed (P5, on data/splits/test.jsonl, n=206)): recall=0.3200, FPR=0.2222, precision=0.6897, F1=0.4372

**By Position** (attack records only):

| Position | Support | Recall | Wilson 95% CI |
|----------|---------|--------|--------------|
| end | 53 | 0.2642 | [0.164, 0.396] |
| middle | 30 | 0.3000 | [0.167, 0.479] |
| start | 42 | 0.4048 | [0.270, 0.555] |

**By Attack Type** (attack records only):

| Attack Type | Support | Recall | Wilson 95% CI |
|-------------|---------|--------|--------------|
| Base Encoding | 5 | 0.0000 | [0.000, 0.434] |
| Blocking Internet Connection | 5 | 0.4000 | [0.118, 0.769] |
| Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.2000 | [0.036, 0.624] |
| Business Intelligence | 5 | 0.2000 | [0.036, 0.624] |
| Compromising Computers | 5 | 0.4000 | [0.118, 0.769] |
| Conversational Agent | 5 | 0.0000 | [0.000, 0.434] |
| Corrupting an Operating System | 5 | 0.2000 | [0.036, 0.624] |
| Data Eavesdropping | 5 | 0.4000 | [0.118, 0.769] |
| Emoji Substitution | 5 | 0.4000 | [0.118, 0.769] |
| Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.6000 | [0.231, 0.882] |
| Entertainment | 5 | 0.0000 | [0.000, 0.434] |
| Information Dissemination | 5 | 0.6000 | [0.231, 0.882] |
| Introduce System Fingerprinting | 5 | 0.2000 | [0.036, 0.624] |
| Keylogging | 5 | 0.4000 | [0.118, 0.769] |
| Language Translation (Test) | 5 | 0.0000 | [0.000, 0.434] |
| Marketing & Advertising | 5 | 0.2000 | [0.036, 0.624] |
| Misinformation & Propaganda | 5 | 0.6000 | [0.231, 0.882] |
| Research Assistance | 5 | 0.6000 | [0.231, 0.882] |
| Reverse Text | 5 | 0.4000 | [0.118, 0.769] |
| Scams & Fraud | 5 | 0.4000 | [0.118, 0.769] |
| Screen Scraping | 5 | 0.6000 | [0.231, 0.882] |
| Sentiment Analysis | 5 | 0.4000 | [0.118, 0.769] |
| Substitution Ciphers | 5 | 0.0000 | [0.000, 0.434] |
| Task Automation | 5 | 0.6000 | [0.231, 0.882] |
| Traffic Analysis | 5 | 0.2000 | [0.036, 0.624] |

---

## 10c. Obfuscation Held-Out Table (team-reported, P3)

> Source: `results/table_3_5_obfuscation_comparison.json` (owner P3, please review)
> Provenance: team-reported. n=206 for each obfuscation type.

| Obfuscation Type | Step1 Recall | Step2 Recall | Step6 Recall |
|-----------------|-------------|-------------|-------------|
| Base64 wrapper | 0.4000 | 0.9520 | 0.9840 |
| Homoglyphs | 0.6320 | 0.6720 | 0.9760 |
| Zero-width characters | 0.4480 | 0.9600 | 0.9840 |
| Leetspeak | 0.6320 | 0.9600 | 0.9920 |
| Markdown embeds | 0.5840 | 0.9600 | 0.9840 |

---

## 10d. Latency Per Stage incl. Model Forward (team-reported, P3)

> Source: `results/table_3_8_latency.json` (owner P3, please review)
> Hardware: AMD Ryzen 7 7735U (16 vCPUs, CPU mode), ~2000-token document

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
|-------|-----------|------------|---------|
| keyword baseline (team) | 43.56 | 41.17 | 55.14 |
| full pipeline / model forward (team) | 166.61 | 167.83 | 185.01 |
| sentence_segmentation (P5-measured) | 19.87 | 17.14 | 33.65 |
| redact (P5-measured) | 0.07 | 0.06 | 0.10 |
| tokenizer_encode (P5-measured) | 3.69 | 3.49 | 4.79 |

---

## 10e. Reconciliation Flags

> P5 recomputes metrics from `raw_test_predictions.jsonl` (is_blocked as predicted label)
> and compares to team-reported step metrics (threshold-calibrated scores).
> Differences expected due to different thresholds; flagged when |diff| > 0.005.

| Step | Metric | Recomputed | Team-Reported | |diff| | Note |
|------|--------|-----------|--------------|-------|------|
| step1 | recall | 0.3200 | 0.5440 | 0.2240 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step2 | recall | 0.3200 | 0.8640 | 0.5440 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step2 | fpr | 0.2222 | 0.5309 | 0.3086 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step3 | recall | 0.3200 | 0.4000 | 0.0800 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step3 | fpr | 0.2222 | 0.3210 | 0.0988 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step4 | recall | 0.3200 | 0.4560 | 0.1360 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step4 | fpr | 0.2222 | 0.2593 | 0.0370 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step5 | recall | 0.3200 | 0.6320 | 0.3120 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step5 | fpr | 0.2222 | 0.3086 | 0.0864 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step6 | recall | 0.3200 | 0.6320 | 0.3120 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |
| step6 | fpr | 0.2222 | 0.2099 | 0.0123 | P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores |

---

## 11. Open Items

| Item | Status | Owner | Note |
|------|--------|-------|------|
| Pair-input collator eval | RESOLVED — step4 metrics present | P3 | results/step4_pair_input/ |
| Obfuscated set (table_3_5) | RESOLVED — table ingested (team-reported) | P3 | results/table_3_5_obfuscation_comparison.json |
| Model-forward latency | RESOLVED — table_3_8 ingested (team-reported) | P3 | results/table_3_8_latency.json |
| Trained token-head weights | BLOCKED — no weight files in repo | P4 | token_head metrics N/A |
| Real LLM ASR | N/A — no LLM endpoint | All | PSR proxy used instead |
| Full splits re-eval (n=206 vs n=42999) | RESOLVED — lexical 5% FPR, primary test n=42999 | P1 | results/calibration_full.json; P5 n=206 stays a separate corpus |
| token_head re-eval | BLOCKED — weights absent | P4 | span-F1 and PSR N/A |

---

## 12. Group Members

| Name | Reg. No. | Email |
|------|----------|-------|
| Kinshuk Gupta | 12341190 | kinshukg@iitbhilai.ac.in |
| Harshit Kandpal | B24CS505 | harshitka@iitbhilai.ac.in |
| Sujal Som | B24DS032 | sujals@iitbhilai.ac.in |
| Utkarsh Chatterjee | B24CS050 | utkarshchatt@iitbhilai.ac.in |
| Tanmay Jaiswal | B24DS033 | tanmay@iitbhilai.ac.in |

---

*Report auto-generated by `scripts/make_p5_report.py`. All numbers injected from `results/p5/*.json`.*
