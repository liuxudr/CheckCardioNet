"""CheckCardioNet — ICI 心血管加速风险预测 CLI.

子命令：
    list-drugs         列出支持的 ICI 药物
    score-patient      单患者预测：CVD 加速评分 + ICI 推荐
    score-cohort       从 CSV 批量预测

工具开箱即用，预训练数据(MR / BDS / dual-benefit atlas)随包分发。
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
        "ICI 心血管加速风险预测 (Predict ICI-driven cardiovascular acceleration "
        "risk and generate individualized ICI recommendations)."
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


# ─── 内部工具：解析表达谱 ─────────────────────────────────────────────────────

def _parse_inline_expr(s: str) -> dict[str, float]:
    """解析 'GENE1=val1,GENE2=val2' 格式的表达谱。"""
    out: dict[str, float] = {}
    if not s:
        return out
    for tok in s.split(","):
        if "=" not in tok:
            raise typer.BadParameter(
                f"--expr 项必须为 'GENE=VALUE' 格式，得到: {tok!r}"
            )
        gene, val = tok.split("=", 1)
        out[gene.strip()] = float(val.strip())
    return out


def _parse_expr_file(path: Path) -> dict[str, float]:
    """从 CSV/TSV 读取两列(gene,value)表达谱。"""
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
        f"无法解析 {path}：需要两列(gene, value) CSV 或 'gene' + 'value/expr' 表头。"
    )


def _resolve_expr(
    expr_inline: str | None,
    expr_file: Path | None,
) -> dict[str, float]:
    """合并 --expr 与 --expr-file，文件优先；缺一也允许（仅基于先验）。"""
    expr: dict[str, float] = {}
    if expr_file:
        if not expr_file.exists():
            raise typer.BadParameter(f"--expr-file {expr_file} 不存在")
        expr.update(_parse_expr_file(expr_file))
    if expr_inline:
        expr.update(_parse_inline_expr(expr_inline))
    return expr


# ─── list-drugs ──────────────────────────────────────────────────────────────

@app.command(name="list-drugs")
def list_drugs():
    """列出支持的 ICI 药物及靶点。"""
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
        help="癌种代码（如 NSCLC, SKCM, BLCA, KIRC, HCC, AML, MDS）",
    ),
    cvd_susceptibility: float = typer.Option(
        0.5, "--cvd-susceptibility", "--cvd",
        min=0.0, max=1.0,
        help="基线 CVD 易感性 ∈ [0,1]（low<0.3, moderate 0.3-0.6, high>0.6）",
    ),
    expr: str = typer.Option(
        "", "--expr", "-e",
        help="检查点表达谱（z-score）：'PDCD1=2.5,CD274=1.8,CD47=3.0'",
    ),
    expr_file: Path = typer.Option(
        None, "--expr-file", "-f",
        help="CSV/TSV：两列(gene, value) 或 'gene'+'value/expr' 表头",
    ),
    drug: str = typer.Option(
        "", "--drug", "-d",
        help="（可选）只评估单个 ICI 药物；省略则比较所有支持的 ICI",
    ),
    candidates: str = typer.Option(
        "", "--candidates",
        help="候选 ICI 列表(逗号分隔)，如 'pembrolizumab,nivolumab,magrolimab'；"
             "省略则使用全部",
    ),
    output_json: Path = typer.Option(
        None, "--output-json", "-o",
        help="（可选）将结果写入 JSON",
    ),
):
    """单患者预测：计算 CVD 加速评分并给出 ICI 个体化推荐。"""
    patient_expr = _resolve_expr(expr, expr_file)
    if not patient_expr:
        console.print(
            "[yellow]⚠ 未提供表达谱(--expr / --expr-file)；将仅基于药物先验给出粗略评分。[/yellow]"
        )

    if drug:
        # 单药模式
        from checkcardionet.scoring import ICIDrivenCVDAccelerationScore

        scorer = ICIDrivenCVDAccelerationScore()
        result = scorer.compute(patient_expr, drug, cvd_susceptibility)
        _print_single_drug(result, cancer_type, cvd_susceptibility)
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
            console.print(f"[dim]→ 已保存 {output_json}[/dim]")
        return

    # 多药对比模式
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
        console.print(f"[dim]→ 已保存 {output_json}[/dim]")


# ─── score-cohort ────────────────────────────────────────────────────────────

@app.command(name="score-cohort")
def score_cohort(
    input_csv: Path = typer.Argument(..., help="队列 CSV（每行一个患者）"),
    output_csv: Path = typer.Option(
        Path("predictions.csv"), "--output", "-o",
        help="输出 CSV：每行 = patient × drug 的评分",
    ),
    candidates: str = typer.Option(
        "", "--candidates",
        help="候选 ICI 列表(逗号分隔)；省略则全部",
    ),
):
    """批量队列预测。

    输入 CSV 必备列：
      patient_id, cancer_type, cvd_susceptibility
    + 任意基因表达列（如 PDCD1, CD274, CD47, …），列名必须是基因 symbol。
    """
    import pandas as pd
    from checkcardionet.scoring import IntegratedRecommendationSystem
    from checkcardionet.data.preprocess import load_checkpoint_panel

    if not input_csv.exists():
        raise typer.BadParameter(f"{input_csv} 不存在")
    df = pd.read_csv(input_csv)

    required = {"patient_id", "cancer_type", "cvd_susceptibility"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"输入 CSV 缺少列: {missing}")

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
                    "✓" if r["drug"] == rec["primary_recommendation"] else ""
                ),
            })

    out_df = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    primary = out_df[out_df["primary_recommendation"] == "✓"]
    console.print(
        f"[green]✓ 队列预测完成[/green]：{df.shape[0]} 患者 × "
        f"{out_df['drug'].nunique()} 药物 = {len(out_df)} 条评分 → {output_csv}"
    )
    if not primary.empty:
        console.print("\n[bold]每患者首选 ICI：[/bold]")
        console.print(
            primary[["patient_id", "cancer_type", "drug",
                     "onco_benefit", "cvd_accel_score", "net_benefit", "category"]]
            .to_string(index=False)
        )


# ─── 输出辅助 ─────────────────────────────────────────────────────────────────

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
        comp.add_column("MR β",        justify="right")
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
    """将 numpy / pandas 标量转为内置类型供 json.dumps。"""
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
