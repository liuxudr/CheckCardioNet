"""综合推荐系统：整合肿瘤获益 + CVD 风险 → 个体化 ICI 方案推荐。

决策矩阵：
                Low CVD Accel Risk   High CVD Accel Risk
High Onco Benefit  Standard ICI        Dual-benefit 药物优先
                                       或 Standard ICI + 心脏保护
Low Onco Benefit   ICI 考虑            避免 ICI
                                       (或 dual-benefit only)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preprocess import PRETRAINED_DIR

logger = logging.getLogger(__name__)

# 肿瘤对 ICI 应答的粗略先验（来自文献 ORR）
ONCO_PRIOR_ORR: dict[str, dict[str, float]] = {
    "pembrolizumab": {"SKCM": 0.45, "NSCLC": 0.35, "BLCA": 0.25, "default": 0.25},
    "nivolumab":     {"SKCM": 0.40, "KIRC": 0.25, "HCC": 0.20, "default": 0.22},
    "ipilimumab":    {"SKCM": 0.20, "default": 0.15},
    "atezolizumab":  {"NSCLC": 0.30, "BLCA": 0.20, "default": 0.18},
    "magrolimab":    {"AML": 0.40, "MDS": 0.35, "default": 0.25},
    "relatlimab":    {"SKCM": 0.35, "default": 0.20},
    "cobolimab":     {"NSCLC": 0.25, "default": 0.18},
}


class IntegratedRecommendationSystem:
    """整合三维评分生成个体化 ICI 推荐。

    用法::

        system = IntegratedRecommendationSystem()
        rec = system.generate_recommendation(
            patient_expr={"PDCD1": 2.5, "CD47": 1.0},
            cancer_type="NSCLC",
            cvd_susceptibility=0.6,
        )
        print(rec["recommendation"])
    """

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        from .cvd_acceleration_score import ICIDrivenCVDAccelerationScore
        self._accel_scorer = ICIDrivenCVDAccelerationScore(self.out_dir)

    # ── 主推荐函数 ─────────────────────────────────────────────────────────────

    def generate_recommendation(
        self,
        patient_expr: dict[str, float],
        cancer_type: str,
        cvd_susceptibility: float = 0.5,
        candidate_drugs: list[str] | None = None,
    ) -> dict:
        """生成个体化 ICI 治疗推荐。

        Parameters
        ----------
        patient_expr : dict[str, float]
            患者肿瘤检查点表达谱（z-score）。
        cancer_type : str
            癌种代码（如 "NSCLC", "SKCM"）。
        cvd_susceptibility : float
            基线 CVD 风险（0–1）。
        candidate_drugs : list[str] | None
            候选 ICI 列表，None 则考虑所有已知 ICI。

        Returns
        -------
        dict  包含 recommendation, decision_matrix, ranked_drugs
        """
        if candidate_drugs is None:
            candidate_drugs = list(ONCO_PRIOR_ORR.keys())

        # 加载 dual-benefit atlas
        atlas = self._load_atlas()

        drug_scores = []
        for drug in candidate_drugs:
            onco_benefit = self._estimate_onco_benefit(drug, cancer_type)
            accel_result = self._accel_scorer.compute(patient_expr, drug, cvd_susceptibility)
            accel_score = accel_result["accel_score"]
            net_benefit = onco_benefit - max(0, accel_score) * cvd_susceptibility

            category = self._classify_drug(onco_benefit, accel_score, atlas, drug)

            drug_scores.append({
                "drug": drug,
                "onco_benefit": onco_benefit,
                "cvd_accel_score": accel_score,
                "net_benefit": net_benefit,
                "category": category,
                "recommendation_priority": self._priority(onco_benefit, accel_score),
            })

        df = pd.DataFrame(drug_scores).sort_values("recommendation_priority", ascending=False)

        # 推荐决策
        primary = df.iloc[0] if not df.empty else None
        rec = self._generate_text(primary, df, cancer_type, cvd_susceptibility)

        return {
            "patient_cancer": cancer_type,
            "cvd_susceptibility": cvd_susceptibility,
            "ranked_drugs": df,
            "primary_recommendation": primary["drug"] if primary is not None else "None",
            "recommendation": rec,
        }

    # ── 内部辅助 ─────────────────────────────────────────────────────────────

    def _estimate_onco_benefit(self, drug: str, cancer_type: str) -> float:
        priors = ONCO_PRIOR_ORR.get(drug, {})
        return float(priors.get(cancer_type, priors.get("default", 0.15)))

    @staticmethod
    def _classify_drug(onco: float, accel: float, atlas: pd.DataFrame, drug: str) -> str:
        if atlas is not None and not atlas.empty and "drug" in atlas.columns:
            match = atlas[atlas["drug"] == drug]
            if not match.empty:
                tier = match.iloc[0].get("recommendation_tier", "C")
                if tier == "D":
                    return "Dual-risk ICI"
        if accel < 0:
            return "CVD-protective ICI"
        elif accel < 0.2:
            return "Standard ICI (low CVD concern)"
        elif onco > 0.3 and accel > 0.3:
            return "High-benefit / High-CVD-risk"
        else:
            return "Use with caution"

    @staticmethod
    def _priority(onco: float, accel: float) -> float:
        """推荐优先级：高获益 + 低 CVD 加速 = 高优先级。"""
        return onco - 0.5 * max(0, accel)

    def _load_atlas(self) -> pd.DataFrame | None:
        path = PRETRAINED_DIR / "dual_benefit_atlas.parquet"
        return pd.read_parquet(path) if path.exists() else None

    @staticmethod
    def _generate_text(
        primary: pd.Series | None,
        df: pd.DataFrame,
        cancer_type: str,
        cvd_susc: float,
    ) -> str:
        risk_level = "高" if cvd_susc > 0.6 else ("中" if cvd_susc > 0.3 else "低")
        lines = [f"== 个体化 ICI 方案推荐 ==",
                 f"癌种: {cancer_type} | 基线 CVD 风险: {risk_level}（{cvd_susc:.2f}）\n"]

        if primary is None:
            lines.append("无可用 ICI 候选。")
            return "\n".join(lines)

        lines.append(f"首选方案: {primary['drug']}")
        lines.append(f"  肿瘤获益估计: {primary['onco_benefit']:.0%}")
        lines.append(f"  CVD 加速评分: {primary['cvd_accel_score']:.3f}")
        lines.append(f"  综合净获益: {primary['net_benefit']:.3f}")
        lines.append(f"  分类: {primary['category']}")

        if cvd_susc > 0.6 and primary["cvd_accel_score"] > 0.3:
            lines.append("\n⚠ 心脏保护建议：")
            lines.append("  • 联合他汀 ± 阿司匹林（斑块稳定）")
            lines.append("  • 基线及每3个月超声心动图")
            lines.append("  • 如 CVD 事件高风险，考虑换用 magrolimab（dual-benefit）")

        dual_benefit = df[df["category"].str.contains("CVD-protective|dual-benefit",
                                                        case=False, na=False)]
        if not dual_benefit.empty:
            lines.append(f"\n双效替代方案: {', '.join(dual_benefit['drug'].tolist())}")

        return "\n".join(lines)

    # ── 批量案例演示 ──────────────────────────────────────────────────────────

    def demo_scenarios(self) -> pd.DataFrame:
        """演示三个典型临床场景。"""
        scenarios = [
            dict(name="NSCLC_low_CVD", patient_expr={"PDCD1": 2.0, "CD47": 0.5},
                 cancer_type="NSCLC", cvd_susceptibility=0.2),
            dict(name="SKCM_high_CVD", patient_expr={"PDCD1": 2.5, "CTLA4": 1.5},
                 cancer_type="SKCM", cvd_susceptibility=0.75),
            dict(name="AML_high_CVD_CD47", patient_expr={"CD47": 3.0, "SIRPA": 1.5},
                 cancer_type="AML", cvd_susceptibility=0.7),
        ]
        rows = []
        for s in scenarios:
            rec = self.generate_recommendation(
                s["patient_expr"], s["cancer_type"], s["cvd_susceptibility"]
            )
            rows.append({
                "scenario": s["name"],
                "primary_recommendation": rec["primary_recommendation"],
                "top3_drugs": ", ".join(rec["ranked_drugs"]["drug"].head(3).tolist()),
                "recommendation_summary": rec["recommendation"],
            })
            logger.info("场景 [%s]:\n%s", s["name"], rec["recommendation"])

        df = pd.DataFrame(rows)
        if self.out_dir is not None:
            df.to_parquet(self.out_dir / "demo_scenarios.parquet", index=False)
        return df
