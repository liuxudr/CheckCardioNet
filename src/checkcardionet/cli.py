"""CheckCardioNet — ICI cardiovascular acceleration risk prediction CLI.

Subcommands:
    list-drugs        List supported ICI drugs and targets
    score-patient     Single-patient CVD-acceleration score + ICI recommendation
    score-cohort      Batch prediction from a cohort CSV

Pre-trained artifacts (MR / BDS / dual-benefit atlas) ship inside the package,
so the tool works out of the box with no external data download.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="checkcardionet",
    help=(
        "Predict ICI-driven cardiovascular acceleration risk and generate "
        "individualized ICI recommendations."
    ),
    add_completion=False,
)
console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ─── Helpers: parse expression profile ───────────────────────────────────────

def _parse_inline_expr(s: str) -> dict[str, float]:
    """Parse a 'GENE1=val1,GENE2=val2' string."""
    out: dict[str, float] = {}
    if not s:
        return out
    for tok in s.split(","):
        if "=" not in tok:
            raise typer.BadParameter(
                f"--expr token must be 'GENE=VALUE', got: {tok!r}"
            )
        gene, val = tok.split("=", 1)
        out[gene.strip()] = float(val.strip())
    return out


def _parse_expr_file(path: Path) -> dict[str, float]:
    """Read a two-column (gene, value) expression CSV/TSV."""
    import pandas as pd

    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    cols = [c.lower() for c in df.columns]
    if "gene" in cols and ("value" in cols or "expr" in cols or "expression" in cols):
        gene_col = df.columns[cols.index("gene")]
        val_col = df.columns[cols.index("value")] if "value" in cols else \
                  (df.columns[cols.index("expr")] if "expr" in cols else
                   df.columns[cols.index("expression")])
        return dict(zip(df[gene_col].astype(str), df[val_col].astype(float)))
    if df.shape[1] >= 2:
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(float)))
    raise typer.BadParameter(
        f"Cannot parse {path}: need a two-column (gene, value) CSV or "
        f"a header row with 'gene' + 'value/expr/expression'."
    )


def _resolve_expr(
    expr_inline: str | None,
    expr_file: Path | None,
) -> dict[str, float]:
    """Merge --expr and --expr-file inputs (file first, inline overrides per gene)."""
    expr: dict[str, float] = {}
    if expr_file:
        if not expr_file.exists():
            raise typer.BadParameter(f"--expr-file {expr_file} does not exist")
        expr.update(_parse_expr_file(expr_file))
    if expr_inline:
        expr.update(_parse_inline_expr(expr_inline))
    return expr


# ─── list-drugs ──────────────────────────────────────────────────────────────

@app.command(name="list-drugs")
def list_drugs():
    """List supported ICI drugs and their molecular targets."""
    from checkcardionet.scoring.cvd_acceleration_score import ICI_CVD_PRIOR

    table = Table(title="Supported ICI drugs", show_lines=False)
    table.add_column("Drug", style="cyan")
    table.add_column("Target(s)", style="green")
    table.add_column("CVD risk prior", style="yellow")
    for drug, info in ICI_CVD_PRIOR.items():
        table.add_row(
            drug,
            ", ".join(info["targets"]),
            f"{info['cvd_risk_prior']:+.2f}",
        )
    console.print(table)
    console.print(
        "\n[dim]Tip:[/dim] negative CVD-risk-prior (e.g. magrolimab) means the "
        "drug may be CVD-protective."
    )


# ─── score-patient ───────────────────────────────────────────────────────────

@app.command(name="score-patient")
def score_patient(
    cancer_type: str = typer.Option(
        ..., "--cancer-type", "-c",
        help="TCGA / cancer code (e.g. NSCLC, SKCM, BLCA, KIRC, HCC, AML, MDS)",
    ),
    cvd_susceptibility: float = typer.Option(
        0.5, "--cvd-susceptibility", "--cvd",
        min=0.0, max=1.0,
        help="Baseline CVD susceptibility in [0,1] (low<0.3, moderate 0.3-0.6, high>0.6)",
    ),
    expr: str = typer.Option(
        "", "--expr", "-e",
        help="Inline checkpoint expression profile (z-score), e.g. 'PDCD1=2.5,CD274=1.8,CD47=3.0'",
    ),
    expr_file: Path = typer.Option(
        None, "--expr-file", "-f",
        help="CSV/TSV with two columns (gene, value) or header 'gene' + 'value/expr/expression'",
    ),
    drug: str = typer.Option(
        "", "--drug", "-d",
        help="(Optional) score a single ICI drug; omit to compare all supported drugs",
    ),
    candidates: str = typer.Option(
        "", "--candidates",
        help="Comma-separated candidate ICI list, e.g. 'pembrolizumab,nivolumab,magrolimab'; "
             "omit to use all supported drugs",
    ),
    output_json: Path = typer.Option(
        None, "--output-json", "-o",
        help="(Optional) write the result to a JSON file",
    ),
):
    """Single-patient prediction: CVD-acceleration score + individualized ICI recommendation."""
    patient_expr = _resolve_expr(expr, expr_file)
    if not patient_expr:
        console.print(
            "[yellow]! No expression profile provided (--expr / --expr-file); "
            "scoring will fall back to drug-level priors only.[/yellow]"
        )

    if drug:
        # Single-drug mode
        from checkcardionet.scoring import ICIDrivenCVDAccelerationScore

        scorer = ICIDrivenCVDAccelerationScore()
        result = scorer.compute(patient_expr, drug, cvd_susceptibility)
        _print_single_drug(result, cancer_type, cvd_susceptibility)
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(_jsonable(result), indent=2, ensure_ascii=False)
            )
            console.print(f"[dim]-> saved {output_json}[/dim]")
        return

    # Multi-drug comparison mode
    from checkcardionet.scoring import IntegratedRecommendationSystem

    rec_system = IntegratedRecommendationSystem()
    candidate_list = (
        [d.strip() for d in candidates.split(",") if d.strip()]
        if candidates else None
    )
    rec = rec_system.generate_recommendation(
        patient_expr=patient_expr,
        cancer_type=cancer_type,
        cvd_susceptibility=cvd_susceptibility,
        candidate_drugs=candidate_list,
    )
    _print_recommendation(rec)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        rec_serial = {k: v for k, v in rec.items() if k != "ranked_drugs"}
        rec_serial["ranked_drugs"] = rec["ranked_drugs"].to_dict("records")
        output_json.write_text(json.dumps(rec_serial, indent=2, ensure_ascii=False))
        console.print(f"[dim]-> saved {output_json}[/dim]")


# ─── score-cohort ────────────────────────────────────────────────────────────

@app.command(name="score-cohort")
def score_cohort(
    input_csv: Path = typer.Argument(..., help="Cohort CSV (one patient per row)"),
    output_csv: Path = typer.Option(
        Path("predictions.csv"), "--output", "-o",
        help="Output CSV (one row per patient x drug)",
    ),
    candidates: str = typer.Option(
        "", "--candidates",
        help="Comma-separated candidate ICI list; omit to use all supported drugs",
    ),
):
    """Batch cohort prediction.

    Required input CSV columns:
      patient_id, cancer_type, cvd_susceptibility
    Plus any number of gene-expression columns (e.g. PDCD1, CD274, CD47, ...).
    Column names must match HGNC checkpoint gene symbols.
    """
    import pandas as pd
    from checkcardionet.scoring import IntegratedRecommendationSystem
    from checkcardionet.data.preprocess import load_checkpoint_panel

    if not input_csv.exists():
        raise typer.BadParameter(f"{input_csv} does not exist")
    df = pd.read_csv(input_csv)

    required = {"patient_id", "cancer_type", "cvd_susceptibility"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Input CSV is missing required column(s): {missing}")

    panel = set(load_checkpoint_panel("all_checkpoints"))
    expr_cols = [c for c in df.columns if c in panel]

    candidate_list = (
        [d.strip() for d in candidates.split(",") if d.strip()]
        if candidates else None
    )

    rec_system = IntegratedRecommendationSystem()
    out_rows: list[dict] = []
    for _, row in df.iterrows():
        patient_expr = {g: float(row[g]) for g in expr_cols if pd.notna(row[g])}
        rec = rec_system.generate_recommendation(
            patient_expr=patient_expr,
            cancer_type=str(row["cancer_type"]),
            cvd_susceptibility=float(row["cvd_susceptibility"]),
            candidate_drugs=candidate_list,
        )
        for _, r in rec["ranked_drugs"].iterrows():
            out_rows.append({
                "patient_id": row["patient_id"],
                "cancer_type": row["cancer_type"],
                "cvd_susceptibility": row["cvd_susceptibility"],
                "drug": r["drug"],
                "onco_benefit": r["onco_benefit"],
                "cvd_accel_score": r["cvd_accel_score"],
                "net_benefit": r["net_benefit"],
                "category": r["category"],
                "recommendation_priority": r["recommendation_priority"],
                "primary_recommendation": (
                    "*" if r["drug"] == rec["primary_recommendation"] else ""
                ),
            })

    out_df = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    primary = out_df[out_df["primary_recommendation"] == "*"]
    console.print(
        f"[green]Cohort prediction complete[/green]: {df.shape[0]} patient(s) x "
        f"{out_df['drug'].nunique()} drug(s) = {len(out_df)} score row(s) -> {output_csv}"
    )
    if not primary.empty:
        console.print("\n[bold]Primary recommendation per patient:[/bold]")
        console.print(
            primary[["patient_id", "cancer_type", "drug",
                     "onco_benefit", "cvd_accel_score", "net_benefit", "category"]]
            .to_string(index=False)
        )


# ─── Output helpers ──────────────────────────────────────────────────────────

def _print_single_drug(result: dict, cancer_type: str, cvd: float) -> None:
    drug = result["drug"]
    accel = result["accel_score"]
    panel = Table(title=f"CVD acceleration: {drug} ({cancer_type}, CVD={cvd:.2f})")
    panel.add_column("Field", style="cyan")
    panel.add_column("Value", style="green")
    panel.add_row("Drug",                drug)
    panel.add_row("Targets",             ", ".join(result["targets"]))
    panel.add_row("Accel score",         f"{accel:+.3f}")
    panel.add_row("Interpretation",      result["interpretation"])
    console.print(panel)

    if result.get("components"):
        comp = Table(title="Per-target decomposition")
        comp.add_column("Target", style="cyan")
        comp.add_column("Expr weight", justify="right")
        comp.add_column("MR beta",     justify="right")
        comp.add_column("BDS",         justify="right")
        comp.add_column("Component",   justify="right", style="yellow")
        for tgt, c in result["components"].items():
            comp.add_row(
                tgt,
                f"{c['expr_weight']:.3f}",
                f"{c['mr_beta']:+.3f}",
                f"{c['bds']:+.3f}",
                f"{c['component_score']:+.3f}",
            )
        console.print(comp)


def _print_recommendation(rec: dict) -> None:
    console.print(rec["recommendation"])
    df = rec["ranked_drugs"].head(10).copy()
    table = Table(title=f"\nRanked ICI drugs (top {len(df)})")
    table.add_column("Drug",          style="cyan")
    table.add_column("Onco benefit",  justify="right")
    table.add_column("CVD accel",     justify="right")
    table.add_column("Net benefit",   justify="right", style="yellow")
    table.add_column("Category",      style="green")
    for _, r in df.iterrows():
        table.add_row(
            r["drug"],
            f"{r['onco_benefit']:.0%}",
            f"{r['cvd_accel_score']:+.3f}",
            f"{r['net_benefit']:+.3f}",
            r["category"],
        )
    console.print(table)


def _jsonable(obj):
    """Convert numpy / pandas scalars into builtin types for json.dumps."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    app()
