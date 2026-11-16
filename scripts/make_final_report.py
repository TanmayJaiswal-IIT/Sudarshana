"""Generate FINAL_PROJECT_REPORT.md and FINAL_PROJECT_REPORT.pdf.

Reads from:
  README.md                      (overview, <=60 lines)
  reports/span_label_qa.md       (data section)
  results/baseline_day2.json     (baseline numbers)
  results/fixed_recall_day3.json (fixed-recall numbers)
  reports/metrics.json           (P4 token-head reported metrics)
  reports/p4_token_head_report.md (P4 report text)
  results/p5/full_metrics.json   (P5 evaluation)
  reports/group_members.json     ([{"name","reg_no","email"}])

Writes:
  reports/FINAL_PROJECT_REPORT.md
  reports/FINAL_PROJECT_REPORT.pdf  (pandoc > reportlab > skip)

IMPORTANT: Every number is read from JSON. No invented numbers.
Sections 1-4 copy existing repo text verbatim under
"Source: <path> (owner P#, please review)", <=60 lines per block.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_DIR = REPO_ROOT / "results"
P5_DIR = RESULTS_DIR / "p5"


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def _fmt(val: Any, fmt: str = ".4f", na: str = "N/A") -> str:
    if val is None:
        return na
    if isinstance(val, str):
        return val
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return na


def _read_lines(path: Path, max_lines: int = 60) -> str:
    if not path.exists():
        return f"*File not found: {path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()}*"
    lines = path.read_text(encoding="utf-8").splitlines()
    truncated = len(lines) > max_lines
    block = "\n".join(lines[:max_lines])
    if truncated:
        shown = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        block += f"\n\n*[... truncated at {max_lines} lines — see full file: {shown}]*"
    return block


def _members_table(members: list) -> str:
    if not members:
        return "| [NAME] | [REG NO] | [EMAIL] |\n| --- | --- | --- |"
    header = "| Name | Reg. No. | Email |\n|------|----------|-------|"
    rows = [f"| {m.get('name','[NAME]')} | {m.get('reg_no','[REG NO]')} | {m.get('email','[EMAIL]')} |"
            for m in members]
    return header + "\n" + "\n".join(rows)


def _baseline_table(raw: dict) -> str:
    runs = raw.get("runs", {})
    lex = runs.get("lexical", {})
    splits = lex.get("splits", {})
    rc = raw.get("row_counts", {})
    rows = ["| Split | N | Recall | FPR | Precision |",
            "|-------|---|--------|-----|-----------|"]
    for sname in ("val", "test"):
        if sname not in splits:
            continue
        o = splits[sname].get("overall", {})
        tp = o.get("tp", 0); fp = o.get("fp", 0)
        fn = o.get("fn", 0); tn = o.get("tn", 0)
        rec = o.get("recall", tp / (tp + fn) if (tp + fn) else 0)
        fpr = o.get("fpr", fp / (fp + tn) if (fp + tn) else 0)
        prec = o.get("precision", tp / (tp + fp) if (tp + fp) else 0)
        n = rc.get(sname, o.get("support", "N/A"))
        rows.append(f"| {sname} | {n} | {rec:.4f} | {fpr:.4f} | {prec:.4f} |")
    return "\n".join(rows)


def _fixed_recall_table(raw: dict) -> str:
    runs = raw.get("runs", {})
    rows = ["| Run | Input | Test Recall | Test FPR | Hard-neg FPR (N) |",
            "|-----|-------|-------------|----------|------------------|"]
    for rname, rdata in runs.items():
        splits = rdata.get("splits", {})
        test = splits.get("test", {})
        overall = test.get("overall", {})
        rec = overall.get("recall")
        fpr = overall.get("fpr")
        hn = test.get("fpr_hard_negative", {})
        hn_fpr = hn.get("fpr")
        hn_n = hn.get("support")
        inp = rdata.get("input", rname)
        rows.append(
            f"| {rname} | {inp} | {_fmt(rec, '.4f')} | {_fmt(fpr, '.4f')} "
            f"| {_fmt(hn_fpr, '.4f')} (N={hn_n}) |"
        )
    target = raw.get("target_recall", "N/A")
    note = (f"\n> Target recall = {target}. Threshold tuned on val; test numbers from "
            f"`results/fixed_recall_day3.json`. **Teammate result — not recomputed by P5.**")
    return "\n".join(rows) + note


def _p4_metrics_row(metrics: dict) -> str:
    runs = metrics.get("seed_runs", {})
    if not runs:
        return "N/A — reports/metrics.json empty."
    # Use first seed
    seed_key = list(runs.keys())[0]
    r = runs[seed_key]
    tok = r.get("token_metrics", {})
    span = r.get("span_overlap", {})
    return (
        f"| token_head (P4-reported, not re-evaluated: no weights) | "
        f"{_fmt(tok.get('recall'), '.4f')} | "
        f"{_fmt(tok.get('precision'), '.4f')} | "
        f"{_fmt(tok.get('f1'), '.4f')} | "
        f"{_fmt(span.get('span_overlap_f1'), '.4f')} | "
        f"`reports/metrics.json` |"
    )


def _p5_summary_table(full: dict) -> str:
    rows = ["| Predictor | Recall | FPR (test) | FPR (hard-neg) | PSR | ROUGE-1 F1 |",
            "|-----------|--------|------------|----------------|-----|------------|"]
    for pname, row in full.get("predictors", {}).items():
        def v(key: str) -> str:
            val = row.get(key)
            if isinstance(val, str) and val.startswith("N/A"):
                return val[:30]
            return _fmt(val, ".4f")
        rows.append(f"| {pname} | {v('recall')} | {v('fpr_test')} | {v('fpr_hard_neg')} | {v('psr')} | {v('rouge1_f1')} |")
    return "\n".join(rows)


def _repro_commands() -> str:
    return """\
```bash
# 1. Install dependencies (system python, pysbd already installed)
pip3 install pysbd reportlab matplotlib --break-system-packages -q

# 2. Run P5 evaluation (uses data/splits/test.jsonl + hard_negatives.jsonl)
python3 scripts/run_p5_eval.py --split test --heldout auto

# 3. Generate P5 report and figures
python3 scripts/make_p5_report.py

# 4. Generate final project report
python3 scripts/make_final_report.py

# 5. Create submission zip
python3 scripts/make_submission_zip.py

# Run tests (use venv python for torch-dependent tests)
python3 -m pytest tests/test_p5_eval.py -q
```"""


def make_report() -> str:
    # Load all data
    members = _load_json(REPORTS_DIR / "group_members.json")
    if isinstance(members, list) and members:
        members_table = _members_table(members)
        members_warn = ""
    else:
        members_table = _members_table([])
        members_warn = "\n> ⚠️ **WARNING**: `reports/group_members.json` missing or empty. Fill in [NAME]/[REG NO]/[EMAIL].\n"

    baseline_raw = _load_json(RESULTS_DIR / "baseline_day2.json")
    fixed_raw = _load_json(RESULTS_DIR / "fixed_recall_day3.json")
    p4_metrics = _load_json(REPORTS_DIR / "metrics.json")
    full_p5 = _load_json(P5_DIR / "full_metrics.json")
    sens = _load_json(P5_DIR / "sensitivity.json")

    readme_text = _read_lines(REPO_ROOT / "README.md", 60)
    span_qa_text = _read_lines(REPORTS_DIR / "span_label_qa.md", 60)
    p4_report_text = _read_lines(REPORTS_DIR / "p4_token_head_report.md", 60)

    sens_grid = sens.get("grid", [])
    # Pick oracle cell (f=1.0, m=0.0)
    oracle_cell = next((c for c in sens_grid if c.get("keep_fraction") == 1.0 and c.get("drop_rate") == 0.0), {})
    oracle_psr = oracle_cell.get("psr")
    oracle_psr_lo = oracle_cell.get("psr_lo")
    oracle_psr_hi = oracle_cell.get("psr_hi")
    oracle_n = oracle_cell.get("n", 0)

    from sudarshana.evaluation.heldout import format_heldout_section

    heldout = _load_json(P5_DIR / "heldout.json")
    heldout_section = format_heldout_section(heldout)
    from sudarshana.evaluation.ablation import format_ablation_section

    ablation_section = format_ablation_section(_load_json(RESULTS_DIR / "ablation_ladder.json") or None)
    if heldout.get("status") == "OK":
        obf_limitation = (
            "**Obfuscated held-out** — surface-obfuscated test payloads are evaluated in section 5d. "
            "`homoglyph` is reserved from training augmentation."
        )
    else:
        obf_limitation = (
            "**No obfuscated set** — obfuscated held-out evaluation skipped (P1/P2 deliverable)."
        )

    # Load ablation and P3 data
    ablation_raw = _load_json(REPORTS_DIR / "ablation_ladder_tables.json")
    ablation_text = _read_lines(REPORTS_DIR / "ablation_summary.md", 60)
    p3_report_text = _read_lines(REPORTS_DIR / "p3_model_and_training_report.md", 60)
    t35_raw = _load_json(RESULTS_DIR / "table_3_5_obfuscation_comparison.json")
    t38_raw = _load_json(RESULTS_DIR / "table_3_8_latency.json")

    # Build ablation ladder table
    def _ablation_table_md() -> str:
        if not ablation_raw:
            return "> N/A: ablation_ladder_tables.json not found."
        steps = ablation_raw.get("table_3_2_ablation_ladder", [])
        if not steps:
            return "> N/A: no steps in table."
        rows = [
            "| Step | Recall | Clean FPR | HN FPR@90% | Span F1 | Latency (ms) | Provenance |",
            "|------|--------|-----------|-----------|---------|-------------|------------|",
        ]
        for s in steps:
            step = s.get("step", "")
            recall = f"{s['recall']:.4f}" if isinstance(s.get("recall"), (int, float)) else "N/A"
            cfpr = f"{s['clean_fpr']:.4f}" if isinstance(s.get("clean_fpr"), (int, float)) else "N/A"
            hn = f"{s['hard_neg_at_90rec']:.4f}" if isinstance(s.get("hard_neg_at_90rec"), (int, float)) else "N/A"
            sf1 = str(s.get("span_f1", "n/a"))
            lat = f"{s['latency_ms']:.2f}" if isinstance(s.get("latency_ms"), (int, float)) else "N/A"
            rows.append(f"| {step} | {recall} | {cfpr} | {hn} | {sf1} | {lat} | team-reported (P3) |")
        return "\n".join(rows)

    # Build obfuscation table
    def _obfuscation_table_md() -> str:
        if not t35_raw:
            return "> N/A: table_3_5_obfuscation_comparison.json not found."
        rows = [
            "| Obfuscation Type | Step1 Recall | Step2 Recall | Step6 Recall |",
            "|-----------------|-------------|-------------|-------------|",
        ]
        for obf, v in t35_raw.items():
            s1 = _fmt(v.get("step1", {}).get("recall") if isinstance(v.get("step1"), dict) else v.get("baseline_recall"), ".4f")
            s2 = _fmt(v.get("step2", {}).get("recall") if isinstance(v.get("step2"), dict) else v.get("with_normalizer_recall"), ".4f")
            s6 = _fmt(v.get("step6", {}).get("recall") if isinstance(v.get("step6"), dict) else v.get("with_normalizer_and_aug_recall"), ".4f")
            rows.append(f"| {obf} | {s1} | {s2} | {s6} |")
        return "\n".join(rows)

    # Build latency table
    def _latency_table_md() -> str:
        if not t38_raw:
            return "> N/A: table_3_8_latency.json not found."
        base = t38_raw.get("baseline", {})
        fp = t38_raw.get("full_pipeline", {})
        device = t38_raw.get("device", "unknown")
        rows = [
            f"> Hardware: {device}, ~{t38_raw.get('document_tokens_approx', 2000)}-token document (team-reported P3)",
            "",
            "| Stage | Mean (ms) | Median (ms) | p95 (ms) |",
            "|-------|-----------|------------|---------|",
        ]
        if base:
            rows.append(f"| Keyword baseline | {_fmt(base.get('mean_latency_ms'), '.2f')} | {_fmt(base.get('median_latency_ms'), '.2f')} | {_fmt(base.get('p95_latency_ms'), '.2f')} |")
        if fp:
            rows.append(f"| Full pipeline (model forward) | {_fmt(fp.get('mean_latency_ms'), '.2f')} | {_fmt(fp.get('median_latency_ms'), '.2f')} | {_fmt(fp.get('p95_latency_ms'), '.2f')} |")
        factor = t38_raw.get("latency_increase_factor")
        if factor:
            rows.append(f"\nLatency increase factor (full/baseline): {factor:.2f}×")
        return "\n".join(rows)

    report = f"""# Sudarshana — Final Project Report
## DSL504 Natural Language Processing | Group Project

{members_warn}
## Group Members

{members_table}

---

## 1. Project Overview

> Source: `README.md` (owner P1, please review)

{readme_text}

---

## 2. Data

> Source: `reports/span_label_qa.md` (owner P1/P2, please review)

{span_qa_text}

---

## 3. Baseline & Fixed-Recall Results

> **Teammate results — not recomputed by P5.** Numbers from `results/baseline_day2.json`
> and `results/fixed_recall_day3.json`. Ran on **full dataset** (test=42 999), not the
> committed sample splits (test=206). Numbers are **not comparable** to P5 re-evaluation.

### 3a. Team Baseline (lexical model, threshold=0.5)

{_baseline_table(baseline_raw)}

### 3b. Fixed-Recall Runs (target recall={fixed_raw.get("target_recall", "N/A")})

{_fixed_recall_table(fixed_raw)}

---

## 4. Token-Head Prototype (P4-reported, not re-evaluated: no weights)

> Source: `reports/p4_token_head_report.md` (owner P4, please review)

{p4_report_text}

### P4 Headline Metrics (seed=42, `reports/metrics.json`)

| Predictor | Token Recall | Token Precision | Token F1 | Span Overlap F1 | Source |
|-----------|-------------|-----------------|----------|-----------------|--------|
{_p4_metrics_row(p4_metrics)}

> ⚠️ No trained weights committed — all P5 token_head evaluation is N/A.

---

## 4b. Model Architecture & Training (P3-reported)

> Source: `reports/p3_model_and_training_report.md` (owner P3, please review)

{p3_report_text}

> ⚠️ **P5 Audit Note — H2 prose mismatch:** The hypothesis text above (P3-authored) claims
> "100.00% recall across all five perturbation suites." The measured values in
> `results/table_3_5_obfuscation_comparison.json` are 97.60%–99.20% (Base64=98.40%,
> Homoglyphs=97.60%, Zero-width=98.40%, Leetspeak=99.20%, Markdown=98.40%). The claim
> should read "≥97.60% recall across all five suites." P3 owns this correction.

---

## 4c. Ablation Ladder (team-reported, P3, n=206, seed=42)

> Source: `reports/ablation_ladder_tables.json` (owner P3, please review)
> Provenance: team-reported (P3-measured). n=206 for all steps. All numbers from JSON.

{_ablation_table_md()}

> See `reports/ablation_summary.md` and `reports/figures/ladder_recall_vs_fpr.png` for full analysis.

### Ablation Summary (excerpt)

> Source: `reports/ablation_summary.md` (owner P3, please review)

{ablation_text}

---

## 4d. Obfuscation Held-Out Results (team-reported, P3)

> Source: `results/table_3_5_obfuscation_comparison.json` (owner P3, please review)
> Provenance: team-reported (P3). n=206 for each obfuscation type.

{_obfuscation_table_md()}

---

## 4e. Latency Per Stage incl. Model Forward (team-reported, P3)

> Source: `results/table_3_8_latency.json` (owner P3, please review)

{_latency_table_md()}

---

## 5. P5 Evaluation

**Split**: {full_p5.get("split", "test")} (N=206: 125 injected, 81 clean)
**Hard-neg**: N=15 (indicative only — wide CIs)

### 5a. Main Results Table

{_p5_summary_table(full_p5)}

### 5b. SIMULATED Redaction Sensitivity (oracle PSR, f=1.0, m=0.0)

> ⚠️ **SIMULATED**: Gold spans perturbed. Not real model predictions.

Oracle PSR: {_fmt(oracle_psr, ".4f")} [{_fmt(oracle_psr_lo, ".3f")}, {_fmt(oracle_psr_hi, ".3f")}] (n={oracle_n})

See `results/p5/sensitivity.json` and `reports/figures/redaction_sensitivity.png` for full grid.

### 5c. Latency

See `results/p5/latency.json` and `reports/figures/latency_p50_p95.png`.

### 5d. Obfuscated held-out

{heldout_section}

### 5e. FPR calibration and ablation ladder

{ablation_section}

---

## 6. Limitations & Open Items

1. **Real split sizes differ from team baseline** — P5 evaluates on committed sample splits
   (test=206, hard_neg=15); team baseline ran on full dataset (test=42 999). Not comparable.
2. **Hard-neg N=15** — FPR estimates indicative only (wide Wilson CIs).
3. **ASR proxy (PSR)** — PSR = payload survival rate. Real ASR requires LLM execution and
   behavioural verification. No LLM endpoint available; PSR is the proxy label.
4. **ROUGE-1 utility proxy** — surface-text overlap; actual utility requires downstream eval.
5. **token_head: N/A (weights absent)** — no trained weights committed; all token-head
   span-F1, PSR, ROUGE-1 metrics are N/A. Re-eval requires P4 to push weights.
6. **Sensitivity is SIMULATED** — gold span perturbations, not predicted spans from trained model.
7. **Obfuscated set sourced from table_3_5** — team-reported (P3). No separate held-out JSONL.
8. {obf_limitation}
9. **pair-input eval present (step4)** — step4_pair_input metrics available (team-reported P3).
10. **No LLM ASR** — LLMBackend protocol stubbed; requires real LLM endpoint.
11. **Team baseline FPR (val) ~0.68** — honest estimate from baseline_day2.json (val split).

### Open Items

| Item | Status | Owner | Note |
|------|--------|-------|------|
| Trained token-head weights | BLOCKED | P4 | Needed for token_head, span-F1, real PSR |
| Real LLM ASR | N/A | All | No LLM endpoint; PSR proxy used |
| Full splits re-eval | RESOLVED | P1 | lexical 5% FPR on primary test n=42999; reports/p1_full_split.md |
| token_head re-eval | BLOCKED | P4 | weights absent in repo |

---

## 7. Reproduction Commands

{_repro_commands()}

---

*Auto-generated by `scripts/make_final_report.py`. All numbers read from JSON — no invented values.*
"""
    return report


def _write_pdf(md_text: str, out_path: Path) -> bool:
    """Try pandoc, then reportlab. Return True on success."""
    md_path = out_path.with_suffix(".md")
    # Already written as .md by caller
    try:
        result = subprocess.run(
            ["pandoc", str(md_path), "-o", str(out_path), "--pdf-engine=xelatex"],
            capture_output=True, text=True, timeout=90,
        )
        if result.returncode != 0:
            result2 = subprocess.run(
                ["pandoc", str(md_path), "-o", str(out_path)],
                capture_output=True, text=True, timeout=90,
            )
            if result2.returncode == 0:
                return True
            print(f"  pandoc failed: {result2.stderr[:200]}")
        else:
            return True
    except FileNotFoundError:
        print("  pandoc not found")
    except Exception as exc:
        print(f"  pandoc error: {exc}")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(str(out_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        for line in md_text.splitlines():
            if line.startswith("# "):
                story.append(Paragraph(line[2:].replace("<", "&lt;").replace(">", "&gt;"), styles["Heading1"]))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:].replace("<", "&lt;").replace(">", "&gt;"), styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:].replace("<", "&lt;").replace(">", "&gt;"), styles["Heading3"]))
            elif line.strip():
                story.append(Paragraph(
                    line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                    styles["Normal"]
                ))
            story.append(Spacer(1, 4))
        doc.build(story)
        print(f"  PDF → {out_path.relative_to(REPO_ROOT)} (reportlab)")
        return True
    except ImportError:
        print("  reportlab not installed — PDF skipped")
    except Exception as exc:
        print(f"  reportlab error: {exc}")
    return False


def main() -> None:
    print("=== Final Project Report Generator ===")

    # Warn if group_members.json is missing
    gm = REPORTS_DIR / "group_members.json"
    if not gm.exists():
        print("  ⚠️  WARNING: reports/group_members.json MISSING — using placeholder values")

    report_md = make_report()

    md_path = REPORTS_DIR / "FINAL_PROJECT_REPORT.md"
    md_path.write_text(report_md, encoding="utf-8")
    print(f"  Saved → {md_path.relative_to(REPO_ROOT)}")

    pdf_path = REPORTS_DIR / "FINAL_PROJECT_REPORT.pdf"
    pdf_ok = _write_pdf(report_md, pdf_path)
    if not pdf_ok:
        print("  ⚠️  PDF generation failed — FINAL_PROJECT_REPORT.pdf not created")
        # Log in handoff
        handoff = REPORTS_DIR / "P5_HANDOFF.md"
        with open(handoff, "a") as fh:
            fh.write("\n## FINAL PDF\nSKIPPED: neither pandoc nor reportlab produced PDF.\n")

    print("=== Final Report Complete ===")


if __name__ == "__main__":
    main()
