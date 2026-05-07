"""ICI 治疗加速 CVD 的风险评分（核心创新）。

公式：
  CVD_accel_score = Σ_{c ∈ targets(ICI)}
      weight_c × MR_cvd_effect_c × patient_susceptibility_c

各分量：
  weight_c           — 该检查点在患者肿瘤中的表达（eQTL/RNA-seq 数据）
  MR_cvd_effect_c    — 该检查点对 CVD 的 MR 因果效应（beta，来自 Phase 4）
  patient_susceptibility_c — 患者基线 CVD 易感性 × 检查点双向性

高分 → ICI 方案对该患者显著加速 CVD 进展的概率高
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preprocess import PRETRAINED_DIR, load_checkpoint_panel

logger = logging.getLogger(__name__)

# 每种 ICI 药物对应的靶点及已知 CVD 风险权重（文献先验）
ICI_CVD_PRIOR: dict[str, dict] = {
    "pembrolizumab": {"targets": ["PDCD1"], "cvd_risk_prior": 0.7},
    "nivolumab":     {"targets": ["PDCD1"], "cvd_risk_prior": 0.7},
    "ipilimumab":    {"targets": ["CTLA4"], "cvd_risk_prior": 0.6},
    "atezolizumab":  {"targets": ["CD274"], "cvd_risk_prior": 0.5},
    "durvalumab":    {"targets": ["CD274"], "cvd_risk_prior": 0.5},
    "relatlimab":    {"targets": ["LAG3"],  "cvd_risk_prior": 0.3},
    "tiragolumab":   {"targets": ["TIGIT"], "cvd_risk_prior": 0.3},
    "cobolimab":     {"targets": ["HAVCR2"],"cvd_risk_prior": 0.5},
    "magrolimab":    {"targets": ["CD47"],  "cvd_risk_prior": -0.3},  # 负值 = CVD 保护
    "monalizumab":   {"targets": ["KLRC1"], "cvd_risk_prior": 0.2},
    "MK-4830":       {"targets": ["LILRB2"],"cvd_risk_prior": 0.2},
}


class ICIDrivenCVDAccelerationScore:
    """评估 ICI 方案对特定患者加速 CVD 进展的风险。

    用法::

        scorer = ICIDrivenCVDAccelerationScore()
        # 患者表达谱（基因表达水平，可来自 RNA-seq）
        patient_expr = {"PDCD1": 2.5, "CD274": 1.8, "CD47": 3.1}
        result = scorer.compute(patient_expr, proposed_ici="pembrolizumab",
                                cvd_susceptibility=0.6)
    """

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._mr_effects = self._load_mr_effects()
        self._bds = self._load_bds()

    def _load_mr_effects(self) -> dict[str, float]:
        """加载 MR 分析中各检查点对 CVD 的平均效应（beta）。"""
        path = PRETRAINED_DIR / "mr_results.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return df.groupby("gene")["ivw_beta"].median().to_dict()

    def _load_bds(self) -> dict[str, float]:
        """加载双向性评分（BDS）。"""
        path = PRETRAINED_DIR / "bidirectional_scores.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return df["BDS"].to_dict()

    # ── 单患者评分 ─────────────────────────────────────────────────────────────

    def compute(
        self,
        patient_expr: dict[str, float],
        proposed_ici: str,
        cvd_susceptibility: float = 0.5,
    ) -> dict:
        """计算单患者的 ICI-CVD 加速评分。

        Parameters
        ----------
        patient_expr : dict[str, float]
            患者肿瘤中检查点基因的表达水平（z-score 或 log-TPM）。
        proposed_ici : str
            拟使用的 ICI 药物名称。
        cvd_susceptibility : float
            患者基线 CVD 易感性（0–1，来自 CVDBaselineRiskScore）。

        Returns
        -------
        dict  含 accel_score, components, interpretation
        """
        ici_info = ICI_CVD_PRIOR.get(proposed_ici, {})
        targets = ici_info.get("targets", [])
        cvd_risk_prior = ici_info.get("cvd_risk_prior", 0.3)

        components = {}
        for target in targets:
            # 患者表达水平（归一化到 0–1）
            expr = patient_expr.get(target, 0.0)
            expr_weight = float(np.clip((expr + 3) / 6, 0, 1))  # 假设 z-score 范围 -3~3

            # MR 效应（数据驱动；无数据则用先验）
            mr_beta = self._mr_effects.get(target, cvd_risk_prior * 0.5)

            # 双向性调节
            bds = self._bds.get(target, 0.0)
            bidirectionality_mod = 1.0 + abs(bds) * 0.3

            comp = expr_weight * abs(mr_beta) * bidirectionality_mod * cvd_susceptibility
            if mr_beta < 0:
                comp = -comp  # CVD 保护性靶点
            components[target] = {
                "expr_weight": expr_weight,
                "mr_beta": mr_beta,
                "bds": bds,
                "component_score": comp,
            }

        accel_score = sum(c["component_score"] for c in components.values())
        accel_score = float(np.clip(accel_score, -1.0, 1.0))

        interpretation = self._interpret(accel_score, proposed_ici)

        return {
            "drug": proposed_ici,
            "targets": targets,
            "accel_score": accel_score,
            "cvd_susceptibility": cvd_susceptibility,
            "components": components,
            "interpretation": interpretation,
        }

    @staticmethod
    def _interpret(score: float, drug: str) -> str:
        if score > 0.4:
            return f"高风险：{drug} 可能显著加速 CVD 进展，建议选择替代方案或加强心脏保护"
        elif score > 0.2:
            return f"中风险：{drug} 可能适度加速 CVD，建议密切心血管监测"
        elif score > 0:
            return f"低风险：{drug} 对 CVD 影响较小，标准心脏监测即可"
        else:
            return f"CVD 保护：{drug} 可能对心血管具有保护作用"

    # ── 批量评分（多药物 × 多患者场景）────────────────────────────────────────

    def batch_score(
        self,
        patient_profiles: pd.DataFrame,
        ici_drugs: list[str] | None = None,
        expr_cols: list[str] | None = None,
        cvd_col: str = "cvd_susceptibility",
    ) -> pd.DataFrame:
        """对多患者 × 多 ICI 方案批量计算 CVD 加速评分。

        Parameters
        ----------
        patient_profiles : pd.DataFrame
            每行一个患者，含表达列和 cvd_susceptibility 列。
        ici_drugs : list[str] | None
            ICI 药物列表，None 则使用全部已知 ICI。
        expr_cols : list[str] | None
            表达列名（基因名），None 则自动检测。

        Returns
        -------
        pd.DataFrame  行=患者, 列=药物 accel_score
        """
        if ici_drugs is None:
            ici_drugs = list(ICI_CVD_PRIOR.keys())
        if expr_cols is None:
            expr_cols = [c for c in patient_profiles.columns
                        if c in load_checkpoint_panel("all_checkpoints")]

        rows = []
        for idx, row in patient_profiles.iterrows():
            expr = {col: row[col] for col in expr_cols if col in row}
            cvd_susc = float(row.get(cvd_col, 0.5))
            patient_scores = {"patient_id": idx}
            for drug in ici_drugs:
                result = self.compute(expr, drug, cvd_susc)
                patient_scores[f"{drug}_accel"] = result["accel_score"]
            rows.append(patient_scores)

        df = pd.DataFrame(rows).set_index("patient_id")
        if self.out_dir is not None:
            df.to_parquet(self.out_dir / "ici_cvd_accel_scores.parquet")
        return df

    # ── 演示案例 ──────────────────────────────────────────────────────────────

    def demo_case_studies(self) -> pd.DataFrame:
        """生成三个典型案例演示。"""
        cases = [
            {
                "name": "Patient A (NSCLC, low CVD risk)",
                "expr": {"PDCD1": 2.0, "CD274": 1.5, "CD47": 0.5, "CTLA4": 0.8},
                "cvd_susceptibility": 0.2,
                "proposed_ici": "pembrolizumab",
            },
            {
                "name": "Patient B (SKCM, high CVD risk, AS history)",
                "expr": {"PDCD1": 2.5, "CD274": 2.0, "CD47": 1.0, "CTLA4": 1.2},
                "cvd_susceptibility": 0.8,
                "proposed_ici": "nivolumab",
            },
            {
                "name": "Patient C (AML, high CVD risk, magrolimab candidate)",
                "expr": {"CD47": 3.0, "SIRPA": 1.5, "MERTK": 1.0},
                "cvd_susceptibility": 0.7,
                "proposed_ici": "magrolimab",
            },
        ]

        rows = []
        for case in cases:
            result = self.compute(
                case["expr"], case["proposed_ici"], case["cvd_susceptibility"]
            )
            rows.append({
                "case": case["name"],
                "proposed_ici": case["proposed_ici"],
                "cvd_susceptibility": case["cvd_susceptibility"],
                "accel_score": result["accel_score"],
                "interpretation": result["interpretation"],
            })

        df = pd.DataFrame(rows)
        if self.out_dir is not None:
            df.to_parquet(self.out_dir / "demo_case_studies.parquet", index=False)

        for _, r in df.iterrows():
            logger.info(
                "[%s]\n  药物=%s, 加速评分=%.3f\n  解读: %s",
                r["case"], r["proposed_ici"], r["accel_score"], r["interpretation"],
            )
        return df
