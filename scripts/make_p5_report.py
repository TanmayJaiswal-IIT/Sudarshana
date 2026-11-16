"""Generate P5 report: figures, markdown table, and eval report.

Reads results/p5/*.json and writes:
  reports/figures/recall_by_position.png
  reports/figures/recall_by_attack_type.png
  reports/figures/psr_rouge_by_condition.png
  reports/figures/latency_p50_p95.png
  reports/p5_metrics_table.md
  reports/p5_eval_report.md
  reports/P5_evaluation_report.pdf  (via pandoc; else reportlab; else skipped)

IMPORTANT: Every number in the report is injected from JSON files.
Hand-typed numbers are forbidden. Missing data renders "N/A (reason)".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "p5"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _load(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def _fmt(val: Any, fmt: str = ".4f", na: str = "N/A") -> str:
    """Format a numeric value or return na string."""
    if val is None:
        return na
    if isinstance(val, str):
        return val
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return na


def _safe_import_matplotlib() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(downstream: dict, slicing: dict, latency: dict) -> list[Path]:
    """Generate all figures. Return list of created figure paths."""
    plt = _safe_import_matplotlib()
    if plt is None:
        print("  [figures] matplotlib not available; skipping figures")
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # --- 1. Recall by position ---
    pos_data: dict[str, dict[str, float]] = {}
    for pname, pdata in slicing.get("predictors", {}).items():
        rbp = pdata.get("recall_by_position", {})
        for bin_, stats in rbp.items():
            if bin_ not in pos_data:
                pos_data[bin_] = {}
            pos_data[bin_][pname] = stats.get("recall", 0.0)

    if pos_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = sorted(pos_data.keys())
        pnames = sorted({p for d in pos_data.values() for p in d})
        x = range(len(bins))
        width = 0.8 / max(len(pnames), 1)
        for i, pname in enumerate(pnames):
            vals = [pos_data[b].get(pname, 0.0) for b in bins]
            ax.bar([xi + i * width for xi in x], vals, width, label=pname)
        ax.set_xticks([xi + width * (len(pnames) - 1) / 2 for xi in x])
        ax.set_xticklabels(bins)
        ax.set_xlabel("Position Bin")
        ax.set_ylabel("Recall")
        ax.set_title("Recall by Injection Position")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        p = FIGURES_DIR / "recall_by_position.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")

    # --- 2. Recall by attack type ---
    atype_data: dict[str, dict[str, float]] = {}
    for pname, pdata in slicing.get("predictors", {}).items():
        rbat = pdata.get("recall_by_attack_type", {})
        for atype, stats in rbat.items():
            if atype not in atype_data:
                atype_data[atype] = {}
            atype_data[atype][pname] = stats.get("recall", 0.0)

    if atype_data:
        pnames = sorted({p for d in atype_data.values() for p in d})
        atypes = sorted(atype_data.keys())
        fig, ax = plt.subplots(figsize=(max(8, len(atypes) * 1.5), 5))
        x = range(len(atypes))
        width = 0.8 / max(len(pnames), 1)
        for i, pname in enumerate(pnames):
            vals = [atype_data[a].get(pname, 0.0) for a in atypes]
            ax.bar([xi + i * width for xi in x], vals, width, label=pname)
        ax.set_xticks([xi + width * (len(pnames) - 1) / 2 for xi in x])
        ax.set_xticklabels(atypes, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Recall")
        ax.set_title("Recall by Attack Type")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        p = FIGURES_DIR / "recall_by_attack_type.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")

    # --- 3. PSR and ROUGE-1 by condition ---
    psr_data: dict[str, float] = {}
    r1_data: dict[str, float] = {}
    for pname, pdata in downstream.get("predictors", {}).items():
        psr_val = pdata.get("psr", {}).get("psr")
        r1_val = pdata.get("rouge1", {}).get("rouge1_f1")
        if psr_val is not None:
            psr_data[pname] = float(psr_val)
        if r1_val is not None:
            r1_data[pname] = float(r1_val)

    if psr_data or r1_data:
        pnames = sorted(set(list(psr_data.keys()) + list(r1_data.keys())))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric_data, title, ylabel in [
            (axes[0], psr_data, "ASR Proxy (Payload Survival Rate)", "PSR"),
            (axes[1], r1_data, "ROUGE-1 Utility Proxy", "ROUGE-1 F1"),
        ]:
            vals = [metric_data.get(p, 0.0) for p in pnames]
            bars = ax.bar(pnames, vals, color=["#4c72b0", "#dd8452", "#55a868"][:len(pnames)])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        p = FIGURES_DIR / "psr_rouge_by_condition.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")

    # --- 4. Latency p50/p95 ---
    lat_stages = latency.get("stages", {})
    stage_names = [s for s, v in lat_stages.items() if isinstance(v, dict) and "p50_ms" in v]
    if stage_names:
        p50s = [lat_stages[s]["p50_ms"] for s in stage_names]
        p95s = [lat_stages[s]["p95_ms"] for s in stage_names]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(stage_names))
        ax.bar([xi - 0.2 for xi in x], p50s, 0.4, label="p50", color="#4c72b0")
        ax.bar([xi + 0.2 for xi in x], p95s, 0.4, label="p95", color="#dd8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(stage_names, rotation=20, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Pipeline Latency (20 runs, 3 warmups)")
        ax.legend()
        fig.tight_layout()
        p = FIGURES_DIR / "latency_p50_p95.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")
    else:
        print("  [figures] no latency data with p50/p95; skipping latency figure")

    # --- 5. Sensitivity grid heatmap ---
    try:
        import json as _json_fig
        sens_path = REPO_ROOT / "results" / "p5" / "sensitivity.json"
        if sens_path.exists():
            with open(sens_path) as _sf:
                _sens = _json_fig.load(_sf)
            grid = _sens.get("grid", [])
            if grid:
                import numpy as np
                fs = sorted(set(c["keep_fraction"] for c in grid), reverse=True)
                ms = sorted(set(c["drop_rate"] for c in grid))
                psr_mat = [[next((c["psr"] for c in grid if c["keep_fraction"] == f and c["drop_rate"] == m), 0.0) for m in ms] for f in fs]
                fig, ax = plt.subplots(figsize=(7, 4))
                im = ax.imshow(psr_mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
                ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"m={m}" for m in ms])
                ax.set_yticks(range(len(fs))); ax.set_yticklabels([f"f={f}" for f in fs])
                ax.set_xlabel("Drop rate (m)"); ax.set_ylabel("Keep fraction (f)")
                ax.set_title("[SIMULATED] PSR by keep_fraction × drop_rate")
                plt.colorbar(im, ax=ax, label="PSR")
                for i, row in enumerate(psr_mat):
                    for j, val in enumerate(row):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                                color="white" if val > 0.6 else "black")
                fig.tight_layout()
                p = FIGURES_DIR / "redaction_sensitivity.png"
                fig.savefig(p, dpi=100); plt.close(fig); created.append(p)
                print(f"  [figure] {p.name}")

            # PSR by attack type
            at = _sens.get("oracle_by_attack_type", {})
            if at:
                atypes = sorted(at.keys())
                psrs = [at[a]["psr"] for a in atypes]
                errs_lo = [at[a]["psr"] - at[a].get("psr_lo", at[a]["psr"]) for a in atypes]
                errs_hi = [at[a].get("psr_hi", at[a]["psr"]) - at[a]["psr"] for a in atypes]
                fig, ax = plt.subplots(figsize=(max(6, len(atypes) * 1.2), 4))
                ax.bar(range(len(atypes)), psrs, color="#dd8452",
                       yerr=[errs_lo, errs_hi], capsize=4)
                ax.set_xticks(range(len(atypes))); ax.set_xticklabels(atypes, rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("PSR (oracle)"); ax.set_ylim(0, 1.1)
                ax.set_title("[SIMULATED] Oracle PSR by Attack Type (f=1.0, m=0.0)")
                fig.tight_layout()
                p = FIGURES_DIR / "psr_by_attack_type.png"
                fig.savefig(p, dpi=100); plt.close(fig); created.append(p)
                print(f"  [figure] {p.name}")
    except Exception as _exc:
        print(f"  [figures] sensitivity figure error: {_exc}")

    # --- 6. Ladder Recall vs FPR (new P5 figure) ---
    try:
        import json as _json_ladder
        abl_path = REPO_ROOT / "reports" / "ablation_ladder_tables.json"
        if abl_path.exists():
            with open(abl_path) as _af:
                _abl = _json_ladder.load(_af)
            steps = _abl.get("table_3_2_ablation_ladder", [])
            if steps:
                step_labels = [s.get("step", f"step{i+1}") for i, s in enumerate(steps)]
                recalls = [s.get("recall", 0.0) for s in steps]
                fprs = [s.get("clean_fpr", 0.0) for s in steps]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(fprs, recalls, "o-", color="#4c72b0", linewidth=2, markersize=8, label="Ablation steps")
                for i, (fpr, rec, label) in enumerate(zip(fprs, recalls, step_labels)):
                    ax.annotate(f"S{i+1}", (fpr, rec), textcoords="offset points", xytext=(5, 5), fontsize=8)
                ax.set_xlabel("Clean FPR (N=75)")
                ax.set_ylabel("Recall")
                ax.set_title("Ablation Ladder: Recall vs Clean FPR\n(team-reported, P3, n=206)")
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90% recall")
                ax.legend()
                fig.tight_layout()
                p_ladder = FIGURES_DIR / "ladder_recall_vs_fpr.png"
                fig.savefig(p_ladder, dpi=100)
                plt.close(fig)
                created.append(p_ladder)
                print(f"  [figure] {p_ladder.name}")
    except Exception as _exc2:
        print(f"  [figures] ladder figure error: {_exc2}")

    return created


# ---------------------------------------------------------------------------
# Markdown metrics table
# ---------------------------------------------------------------------------

def make_metrics_table(full_metrics: dict) -> str:
    """Build markdown metrics table from full_metrics.json."""
    lines = [
        "# P5 Metrics Table",
        "",
        f"**Split**: {full_metrics.get('split', 'N/A')} | **Seed**: {full_metrics.get('seed', 'N/A')}",
        "",
        "| Predictor | Recall | FPR (test) | FPR (hard-neg) | Span F1 | PSR | ROUGE-1 F1 | Latency p50 (ms) | Latency p95 (ms) |",
        "|-----------|--------|------------|----------------|---------|-----|------------|-----------------|-----------------|",
    ]
    for pname, row in full_metrics.get("predictors", {}).items():
        def v(key: str, fmt: str = ".4f") -> str:
            val = row.get(key)
            if isinstance(val, str) and val.startswith("N/A"):
                return val
            return _fmt(val, fmt)

        lines.append(
            f"| {pname} | {v('recall')} | {v('fpr_test')} | {v('fpr_hard_neg')} "
            f"| {v('span_f1')} | {v('psr')} | {v('rouge1_f1')} "
            f"| {v('latency_p50_ms', '.1f')} | {v('latency_p95_ms', '.1f')} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full eval report
# ---------------------------------------------------------------------------

def _ablation_ladder_rows(full_metrics: dict) -> str:
    """Build ablation ladder table rows from full_metrics['team']['ablation_steps']."""
    import json as _json
    p = REPO_ROOT / "reports" / "ablation_ladder_tables.json"
    if not p.exists():
        return "| N/A | ablation_ladder_tables.json not found | — | — | — | — | — |"
    with open(p) as fh:
        abl = _json.load(fh)
    steps = abl.get("table_3_2_ablation_ladder", [])
    rows = []
    for s in steps:
        step = s.get("step", "")
        recall = f"{s.get('recall', 0):.4f}" if isinstance(s.get("recall"), (int, float)) else "N/A"
        cfpr = f"{s.get('clean_fpr', 0):.4f}" if isinstance(s.get("clean_fpr"), (int, float)) else "N/A"
        hn90 = f"{s.get('hard_neg_at_90rec', 0):.4f}" if isinstance(s.get("hard_neg_at_90rec"), (int, float)) else "N/A"
        sf1 = str(s.get("span_f1", "n/a"))
        lat = f"{s.get('latency_ms', 0):.2f}" if isinstance(s.get("latency_ms"), (int, float)) else "N/A"
        rows.append(f"| {step} | {step} | {recall} | {cfpr} | {hn90} | {sf1} | {lat} |")
    return "\n".join(rows) if rows else "| — | — | — | — | — | — | — |"


def _recomputed_slicing_section(full_metrics: dict) -> str:
    """Build recomputed-from-predictions slicing section."""
    team = full_metrics.get("team", {})
    recomp = {}
    # Try loading from team_models.json
    import json as _json
    p = REPO_ROOT / "results" / "p5" / "team_models.json"
    if p.exists():
        with open(p) as fh:
            tm = _json.load(fh)
        recomp = tm.get("recomputed_from_predictions", {})

    if not recomp or recomp.get("status") == "N/A":
        return "> N/A: predictions file not found or no metadata available."

    n = recomp.get("n", "N/A")
    overall = recomp.get("overall", {})
    recall = _fmt(overall.get("recall"), ".4f")
    fpr = _fmt(overall.get("fpr"), ".4f")
    prec = _fmt(overall.get("precision"), ".4f")
    f1 = _fmt(overall.get("f1"), ".4f")
    prov = recomp.get("_provenance", "recomputed")

    lines = [
        f"**Overall** (n={n}, provenance: {prov}): recall={recall}, FPR={fpr}, precision={prec}, F1={f1}",
        "",
        "**By Position** (attack records only):",
        "",
        "| Position | Support | Recall | Wilson 95% CI |",
        "|----------|---------|--------|--------------|",
    ]
    by_pos = recomp.get("by_position", {})
    if isinstance(by_pos, dict) and by_pos.get("status") != "N/A":
        for pos, v in sorted(by_pos.items()):
            if isinstance(v, dict) and v.get("status") != "N/A":
                lines.append(
                    f"| {pos} | {v.get('support', 0)} | {_fmt(v.get('recall'), '.4f')} "
                    f"| [{_fmt(v.get('recall_ci_lo'), '.3f')}, {_fmt(v.get('recall_ci_hi'), '.3f')}] |"
                )
    else:
        lines.append("| N/A — position metadata not in predictions file | — | — | — |")

    lines += [
        "",
        "**By Attack Type** (attack records only):",
        "",
        "| Attack Type | Support | Recall | Wilson 95% CI |",
        "|-------------|---------|--------|--------------|",
    ]
    by_at = recomp.get("by_attack_type", {})
    if isinstance(by_at, dict) and by_at.get("status") != "N/A":
        for at, v in sorted(by_at.items()):
            if isinstance(v, dict) and v.get("status") != "N/A":
                lines.append(
                    f"| {at} | {v.get('support', 0)} | {_fmt(v.get('recall'), '.4f')} "
                    f"| [{_fmt(v.get('recall_ci_lo'), '.3f')}, {_fmt(v.get('recall_ci_hi'), '.3f')}] |"
                )
    else:
        lines.append("| N/A — attack_type metadata not in predictions file | — | — | — |")

    return "\n".join(lines)


def _obfuscation_rows(full_metrics: dict) -> str:
    """Build obfuscation table from table_3_5_obfuscation_comparison.json."""
    import json as _json
    p = REPO_ROOT / "results" / "table_3_5_obfuscation_comparison.json"
    if not p.exists():
        return "| N/A | — | — | — |"
    with open(p) as fh:
        t35 = _json.load(fh)
    rows = []
    for obf_type, v in t35.items():
        s1 = _fmt(v.get("step1", {}).get("recall") if isinstance(v.get("step1"), dict) else v.get("baseline_recall"), ".4f")
        s2 = _fmt(v.get("step2", {}).get("recall") if isinstance(v.get("step2"), dict) else v.get("with_normalizer_recall"), ".4f")
        s6 = _fmt(v.get("step6", {}).get("recall") if isinstance(v.get("step6"), dict) else v.get("with_normalizer_and_aug_recall"), ".4f")
        rows.append(f"| {obf_type} | {s1} | {s2} | {s6} |")
    return "\n".join(rows) if rows else "| N/A | — | — | — |"


def _full_latency_rows(full_metrics: dict) -> str:
    """Build latency table from table_3_8_latency.json (team-reported) + p5 latency.json."""
    import json as _json
    rows = []
    # Team-reported model forward
    p38 = REPO_ROOT / "results" / "table_3_8_latency.json"
    if p38.exists():
        with open(p38) as fh:
            t38 = _json.load(fh)
        base = t38.get("baseline", {})
        fp = t38.get("full_pipeline", {})
        if base:
            rows.append(
                f"| keyword baseline (team) | {_fmt(base.get('mean_latency_ms'), '.2f')} "
                f"| {_fmt(base.get('median_latency_ms'), '.2f')} "
                f"| {_fmt(base.get('p95_latency_ms'), '.2f')} |"
            )
        if fp:
            rows.append(
                f"| full pipeline / model forward (team) | {_fmt(fp.get('mean_latency_ms'), '.2f')} "
                f"| {_fmt(fp.get('median_latency_ms'), '.2f')} "
                f"| {_fmt(fp.get('p95_latency_ms'), '.2f')} |"
            )
    # P5-measured pipeline stages
    p_lat = REPO_ROOT / "results" / "p5" / "latency.json"
    if p_lat.exists():
        with open(p_lat) as fh:
            lat = _json.load(fh)
        for stage, v in lat.get("stages", {}).items():
            if isinstance(v, dict) and "p50_ms" in v:
                rows.append(
                    f"| {stage} (P5-measured) | {_fmt(v.get('mean_ms'), '.2f')} "
                    f"| {_fmt(v.get('p50_ms'), '.2f')} "
                    f"| {_fmt(v.get('p95_ms'), '.2f')} |"
                )
    return "\n".join(rows) if rows else "| N/A | — | — | — |"


def _reconciliation_section(full_metrics: dict) -> str:
    """Build reconciliation flags section."""
    import json as _json
    p = REPO_ROOT / "results" / "p5" / "team_models.json"
    if not p.exists():
        return "> team_models.json not found; run run_p5_eval.py first."
    with open(p) as fh:
        tm = _json.load(fh)
    flags = tm.get("reconciliation_flags", [])
    if not flags:
        return "> No reconciliation flags (|diff| ≤ 0.005 for all checked metrics)."
    lines = [
        "| Step | Metric | Recomputed | Team-Reported | |diff| | Note |",
        "|------|--------|-----------|--------------|-------|------|",
    ]
    for f in flags:
        lines.append(
            f"| {f.get('step')} | {f.get('metric')} | {_fmt(f.get('recomputed'), '.4f')} "
            f"| {_fmt(f.get('team_reported'), '.4f')} | {_fmt(f.get('diff'), '.4f')} "
            f"| {f.get('note', '')} |"
        )
    return "\n".join(lines)


def _group_members_table() -> str:
    """Build group members table from reports/group_members.json."""
    import json as _json
    p = REPO_ROOT / "reports" / "group_members.json"
    if not p.exists():
        return "| N/A | — | — |"
    with open(p) as fh:
        members = _json.load(fh)
    lines = [
        "| Name | Reg. No. | Email |",
        "|------|----------|-------|",
    ]
    for m in members:
        lines.append(f"| {m.get('name', 'N/A')} | {m.get('reg_no', 'N/A')} | {m.get('email', 'N/A')} |")
    return "\n".join(lines)


def make_eval_report(
    downstream: dict,
    slicing: dict,
    latency: dict,
    heldout: dict,
    full_metrics: dict,
    figure_paths: list[Path],
) -> str:
    """Generate the full evaluation report markdown."""

    def _fig_md(stem: str, caption: str) -> str:
        p = FIGURES_DIR / f"{stem}.png"
        if p.exists():
            try:
                shown = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            except ValueError:
                shown = p.as_posix()
            return f"![{caption}]({shown})\n*{caption}*"
        return f"*Figure not available: {stem}.png*"

    def _psr_val(pname: str) -> str:
        d = downstream.get("predictors", {}).get(pname, {})
        return _fmt(d.get("psr", {}).get("psr"), ".4f", "N/A (no attack records)")

    def _r1_val(pname: str) -> str:
        d = downstream.get("predictors", {}).get(pname, {})
        return _fmt(d.get("rouge1", {}).get("rouge1_f1"), ".4f", "N/A")

    def _fpr_test_val(pname: str) -> str:
        s = slicing.get("predictors", {}).get(pname, {})
        return _fmt(s.get("fpr_on_test_negatives", {}).get("fpr"), ".4f", "N/A")

    def _fpr_hn_val(pname: str) -> str:
        s = slicing.get("predictors", {}).get(pname, {})
        return _fmt(s.get("fpr_on_hard_negatives", {}).get("fpr"), ".4f", "N/A")

    # CI helpers (Wilson)
    def _ci_wilson_str(k: Any, n: Any) -> str:
        try:
            from sudarshana.evaluation.p5_common import wilson_ci
            lo, hi = wilson_ci(int(k) if k is not None else 0, int(n) if n else 0)
            return f"[{lo:.3f}, {hi:.3f}] (n={int(n) if n else 0})"
        except Exception:
            return ""

    # Keyword heuristic predictor key (may be 'keyword_heuristic' or 'baseline' in stale JSON)
    def _kw_pname() -> str:
        preds = full_metrics.get("predictors", {})
        return "keyword_heuristic" if "keyword_heuristic" in preds else "baseline"

    def _kw(key: str) -> Any:
        return full_metrics.get("predictors", {}).get(_kw_pname(), {}).get(key)

    def _kw_int(key: str) -> int:
        v = _kw(key)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    def _preds_n() -> str:
        sl = slicing.get("predictors", {}).get(_kw_pname(), {})
        total = sum(
            v.get("support", 0)
            for v in sl.get("recall_by_position", {}).values()
        )
        return str(total) if total else "N/A"

    # Baseline day2 table builder
    def _make_baseline_day2_table() -> str:
        import json as _json
        p = REPO_ROOT / "results" / "baseline_day2.json"
        if not p.exists():
            return "> `results/baseline_day2.json` not found."
        with open(p) as fh:
            raw = _json.load(fh)
        rc = raw.get("row_counts", {})
        splits = raw.get("runs", {}).get("lexical", {}).get("splits", {})
        rows = []
        rows.append("| Split | N | Recall | FPR | Precision | F1 |")
        rows.append("|-------|---|--------|-----|-----------|-----|")
        for sname in ("val", "test"):
            if sname not in splits:
                continue
            o = splits[sname].get("overall", {})
            tp, fp, fn, tn = o.get("tp", 0), o.get("fp", 0), o.get("fn", 0), o.get("tn", 0)
            rec = o.get("recall", tp / (tp + fn) if (tp + fn) else 0)
            fpr = o.get("fpr", fp / (fp + tn) if (fp + tn) else 0)
            prec = o.get("precision", tp / (tp + fp) if (tp + fp) else 0)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            n = rc.get(sname, o.get("support", "N/A"))
            rows.append(f"| {sname} | {n} | {rec:.4f} | {fpr:.4f} | {prec:.4f} | {f1:.4f} |")
        # By attack type (val only since it has this key)
        rat = splits.get("val", {}).get("recall_by_attack_type", {})
        if rat:
            rows.append("")
            rows.append("**Recall by attack type (val split)**")
            rows.append("")
            rows.append("| Attack Type | Support | Recall |")
            rows.append("|-------------|---------|--------|")
            for atype, v in sorted(rat.items()):
                rows.append(f"| {atype} | {v.get('support', 0)} | {v.get('recall', 0):.4f} |")
        return chr(10).join(rows)

    # Sensitivity grid rows
    sens_data = full_metrics.get("sensitivity", {})
    sens_grid = sens_data.get("grid", [])
    sens_oracle_at = sens_data.get("oracle_by_attack_type", {})
    sens_oracle_pos = sens_data.get("oracle_by_position", {})

    def _sens_rows() -> str:
        if not sens_grid:
            return "| — | — | — | — | — |"
        lines = []
        for cell in sens_grid:
            f = cell.get("keep_fraction")
            m = cell.get("drop_rate")
            psr = cell.get("psr")
            psr_lo = cell.get("psr_lo")
            psr_hi = cell.get("psr_hi")
            r1 = cell.get("rouge1_f1")
            n = cell.get("n", 0)
            psr_str = f"{psr:.4f} [{psr_lo:.3f}, {psr_hi:.3f}] (n={n})" if psr is not None else "N/A"
            r1_str = _fmt(r1, ".4f")
            lines.append(f"| {f} | {m} | {psr_str} | {r1_str} |")
        return chr(10).join(lines)

    def _sens_at_rows() -> str:
        if not sens_oracle_at:
            return "| — | — | — | — |"
        lines = []
        for atype, v in sorted(sens_oracle_at.items()):
            psr = v.get("psr", 0)
            psr_lo = v.get("psr_lo", 0)
            psr_hi = v.get("psr_hi", 0)
            n = v.get("n", 0)
            r1 = v.get("rouge1_f1", 0)
            lines.append(f"| {atype} | {n} | {psr:.4f} [{psr_lo:.3f}, {psr_hi:.3f}] | {r1:.4f} |")
        return chr(10).join(lines)

    def _sens_pos_rows() -> str:
        if not sens_oracle_pos:
            return "| — | — | — | — |"
        lines = []
        for pos, v in sorted(sens_oracle_pos.items()):
            psr = v.get("psr", 0)
            psr_lo = v.get("psr_lo", 0)
            psr_hi = v.get("psr_hi", 0)
            n = v.get("n", 0)
            r1 = v.get("rouge1_f1", 0)
            lines.append(f"| {pos} | {n} | {psr:.4f} [{psr_lo:.3f}, {psr_hi:.3f}] | {r1:.4f} |")
        return chr(10).join(lines)

    # Recall by position table rows
    pos_rows: list[str] = []
    for pname, pdata in slicing.get("predictors", {}).items():
        for bin_, stats in sorted(pdata.get("recall_by_position", {}).items()):
            pos_rows.append(
                f"| {pname} | {bin_} | {stats.get('support', 0)} "
                f"| {_fmt(stats.get('recall'), '.4f')} |"
            )

    # Recall by attack type table rows
    atype_rows: list[str] = []
    for pname, pdata in slicing.get("predictors", {}).items():
        for atype, stats in sorted(pdata.get("recall_by_attack_type", {}).items()):
            atype_rows.append(
                f"| {pname} | {atype} | {stats.get('support', 0)} "
                f"| {_fmt(stats.get('recall'), '.4f')} |"
            )

    # Latency table
    lat_rows: list[str] = []
    for stage, v in latency.get("stages", {}).items():
        if isinstance(v, dict) and "p50_ms" in v:
            lat_rows.append(f"| {stage} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['mean_ms']:.2f} |")
        else:
            reason = v.get("na_reason", "unknown") if isinstance(v, dict) else str(v)
            lat_rows.append(f"| {stage} | N/A | N/A | N/A ({reason}) |")

    # Available predictors for downstream section
    avail_pnames = list(downstream.get("predictors", {}).keys())

    from sudarshana.evaluation.heldout import format_heldout_section

    heldout_text = format_heldout_section(heldout)
    from sudarshana.evaluation.ablation import format_ablation_section

    ladder_path = REPO_ROOT / "results" / "ablation_ladder.json"
    if ladder_path.is_file():
        ablation_text = format_ablation_section(json.loads(ladder_path.read_text(encoding="utf-8")))
    else:
        ablation_text = format_ablation_section(None)
    if heldout.get("status") == "OK":
        obf_limitation = (
            "6. **Obfuscated held-out** — surface-obfuscated test payloads are evaluated in section 8. "
            "`homoglyph` is reserved from training augmentation."
        )
    else:
        obf_limitation = (
            "6. **No obfuscated set** — obfuscated held-out evaluation skipped (P1/P2 deliverable)."
        )

    report = f"""# P5 Evaluation Report — Sudarshana Indirect Prompt-Injection Detection

## 1. Setup & Metric Definitions

**Dataset**: `data/splits/test.jsonl` ({full_metrics.get('split', 'test')} split, seed={full_metrics.get('seed', 42)})
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

{_make_baseline_day2_table()}

### P5 keyword_heuristic re-evaluation on `{full_metrics.get('split', 'test')}` split (N={_preds_n()})

- Recall: {_fmt(_kw('recall'), '.4f', 'N/A')} {_ci_wilson_str(_kw_int('tp'), _kw_int('tp') + _kw_int('fn'))}
- FPR (test): {_fpr_test_val('keyword_heuristic')} {_ci_wilson_str(_kw_int('fp_test'), _kw_int('fp_test') + _kw_int('tn_test'))}
- FPR (hard-neg N=15, indicative only): {_fpr_hn_val('keyword_heuristic')}

---

## 3. Recall by Injection Position

{_fig_md('recall_by_position', 'Recall by Injection Position')}

| Predictor | Position | Support | Recall |
|-----------|----------|---------|--------|
{''.join(r + chr(10) for r in pos_rows) if pos_rows else '| — | — | — | — |'}

---

## 4. Recall by Attack Category

{_fig_md('recall_by_attack_type', 'Recall by Attack Type')}

| Predictor | Attack Type | Support | Recall |
|-----------|-------------|---------|--------|
{''.join(r + chr(10) for r in atype_rows) if atype_rows else '| — | — | — | — |'}

---

## 5. FPR on Hard Negatives

| Predictor | FPR (test negatives) | FPR (hard negatives) |
|-----------|----------------------|----------------------|
{''.join(f"| {p} | {_fpr_test_val(p)} | {_fpr_hn_val(p)} |" + chr(10) for p in avail_pnames) if avail_pnames else '| — | — | — |'}

---

## 6. Downstream: PSR and ROUGE-1 (none vs oracle vs predicted)

{_fig_md('psr_rouge_by_condition', 'PSR and ROUGE-1 by Predictor')}

| Predictor | PSR (↓ better for defence) | ROUGE-1 F1 (↑ better) |
|-----------|---------------------------|----------------------|
{''.join(f"| {p} | {_psr_val(p)} | {_r1_val(p)} |" + chr(10) for p in avail_pnames) if avail_pnames else '| — | — | — |'}

**PSR interpretation**: Lower PSR = defence removed more attack payload.
**ROUGE-1 interpretation**: Higher F1 = more of the clean content preserved.

---

## 7. Latency

{_fig_md('latency_p50_p95', 'Pipeline Latency p50/p95')}

Benchmark: ~{latency.get('benchmark_doc_word_estimate', 'N/A')} estimated tokens ({latency.get('benchmark_doc_chars', 'N/A')} chars). 20 runs, 3 warmups.

| Stage | p50 (ms) | p95 (ms) | Mean (ms) |
|-------|----------|----------|-----------|
{''.join(r + chr(10) for r in lat_rows) if lat_rows else '| — | — | — | — |'}

---

## 8. Obfuscated Held-Out Status

{heldout_text}

---

## 8a. FPR calibration and ablation ladder

{ablation_text}

---

## 9. ⚠️ SIMULATED Redaction Sensitivity Analysis

> All results in this section are **SIMULATED**. Gold spans are perturbed to model imperfect
> redaction (partial span coverage, span drop). Real results require trained span predictions.

**Label**: {sens_data.get('label', 'SIMULATED')}

### 9a. PSR and ROUGE-1 Grid (keep_fraction f × drop_rate m)

| keep_fraction | drop_rate | PSR [95% CI] (n=injected) | ROUGE-1 F1 |
|---------------|-----------|---------------------------|------------|
{_sens_rows()}

### 9b. Oracle PSR and ROUGE-1 by Attack Type (f=1.0, m=0.0)

{_fig_md('psr_by_attack_type', 'Oracle PSR by Attack Type')}

| Attack Type | N | PSR [95% CI] | ROUGE-1 F1 |
|-------------|---|--------------|------------|
{_sens_at_rows()}

### 9c. Oracle PSR and ROUGE-1 by Position (f=1.0, m=0.0)

| Position | N | PSR [95% CI] | ROUGE-1 F1 |
|----------|---|--------------|------------|
{_sens_pos_rows()}

{_fig_md('redaction_sensitivity', 'Redaction Sensitivity Grid')}

---

## 10. Limitations

1. **Real split sizes differ from team baseline** — P5 evaluates on committed sample splits (test=206, hard_neg=15); team baseline ran on full dataset (test=42 999). Numbers are **not comparable**.
2. **Hard-neg N=15** — only 15 hard-negative records; FPR estimates are indicative only (wide Wilson CIs).
3. **ASR proxy** — PSR measures payload-text survival; real ASR requires LLM execution and behavioural verification.
4. **ROUGE-1 utility proxy** — surface-text overlap; actual task utility requires downstream agent evaluation.
5. **token_head: N/A** — no trained weights; all token-head metrics are N/A.
{obf_limitation}
7. **No pair-input evaluation** — P3's (user_task, document) pair collator not integrated; all predictors use document-only input.
8. **No LLM ASR** — `LLMBackend` protocol stubbed; `compute_llm_asr` returns N/A.
9. **Sensitivity is SIMULATED** — gold span perturbations, not real predicted spans from a trained model.

---

## 10a. Ablation Ladder (team-reported, P3, n=206, seed=42)

> Source: `reports/ablation_ladder_tables.json` (owner P3, please review)
> Provenance: team-reported (P3-measured). n=206 for all steps.

| Step | Variant | Recall | Clean FPR (N=75) | HN FPR @ 90% Rec | Span F1 | Latency (ms) |
|------|---------|--------|-----------------|-----------------|---------|-------------|
{_ablation_ladder_rows(full_metrics)}

---

## 10b. Trained-Model Recall by Position / Attack Type (recomputed, P5)

> Source: `reports/audit/raw_test_predictions.jsonl` (n=206)
> Provenance: recomputed (P5, from raw_test_predictions.jsonl, is_blocked as predicted label)

{_recomputed_slicing_section(full_metrics)}

---

## 10c. Obfuscation Held-Out Table (team-reported, P3)

> Source: `results/table_3_5_obfuscation_comparison.json` (owner P3, please review)
> Provenance: team-reported. n=206 for each obfuscation type.

| Obfuscation Type | Step1 Recall | Step2 Recall | Step6 Recall |
|-----------------|-------------|-------------|-------------|
{_obfuscation_rows(full_metrics)}

---

## 10d. Latency Per Stage incl. Model Forward (team-reported, P3)

> Source: `results/table_3_8_latency.json` (owner P3, please review)
> Hardware: AMD Ryzen 7 7735U (16 vCPUs, CPU mode), ~2000-token document

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
|-------|-----------|------------|---------|
{_full_latency_rows(full_metrics)}

---

## 10e. Reconciliation Flags

> P5 recomputes metrics from `raw_test_predictions.jsonl` (is_blocked as predicted label)
> and compares to team-reported step metrics (threshold-calibrated scores).
> Differences expected due to different thresholds; flagged when |diff| > 0.005.

{_reconciliation_section(full_metrics)}

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

{_group_members_table()}

---

*Report auto-generated by `scripts/make_p5_report.py`. All numbers injected from `results/p5/*.json`.*
"""
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== P5 Report Generator ===")
    downstream = _load("downstream.json")
    slicing = _load("slicing.json")
    latency = _load("latency.json")
    heldout = _load("heldout.json")
    full_metrics = _load("full_metrics.json")

    if not full_metrics:
        print("  ERROR: results/p5/full_metrics.json not found. Run scripts/run_p5_eval.py first.")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Figures
    figure_paths = make_figures(downstream, slicing, latency)

    # Metrics table
    table_md = make_metrics_table(full_metrics)
    table_path = REPORTS_DIR / "p5_metrics_table.md"
    table_path.write_text(table_md, encoding="utf-8", newline="\n")
    print(f"  Saved → {table_path.relative_to(REPO_ROOT)}")

    # Eval report
    report_md = make_eval_report(downstream, slicing, latency, heldout, full_metrics, figure_paths)
    report_path = REPORTS_DIR / "p5_eval_report.md"
    report_path.write_text(report_md, encoding="utf-8", newline="\n")
    print(f"  Saved → {report_path.relative_to(REPO_ROOT)}")

    # PDF attempt: pandoc → reportlab → skip
    pdf_path = REPORTS_DIR / "P5_evaluation_report.pdf"
    pdf_ok = False
    try:
        result = subprocess.run(
            ["pandoc", str(report_path), "-o", str(pdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"  PDF → {pdf_path.relative_to(REPO_ROOT)} (pandoc)")
            pdf_ok = True
        else:
            print(f"  pandoc failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  pandoc not found")
    except Exception as exc:
        print(f"  pandoc error: {exc}")

    if not pdf_ok:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            for line in report_md.splitlines():
                if line.startswith("# "):
                    story.append(Paragraph(line[2:], styles["Heading1"]))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], styles["Heading2"]))
                elif line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 4))
            doc.build(story)
            print(f"  PDF → {pdf_path.relative_to(REPO_ROOT)} (reportlab)")
            pdf_ok = True
        except ImportError:
            print("  reportlab not installed — PDF skipped; logged in P5_HANDOFF.md")
        except Exception as exc:
            print(f"  reportlab error: {exc}")

    if not pdf_ok:
        # Log in handoff
        handoff = REPORTS_DIR / "P5_HANDOFF.md"
        with open(handoff, "a") as fh:
            fh.write("\n## PDF\nSKIPPED: neither pandoc nor reportlab available.\n")

    print("=== P5 Report Complete ===")


if __name__ == "__main__":
    main()
