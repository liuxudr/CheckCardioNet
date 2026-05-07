"""Integrated recommendation system: tumor benefit + CVD risk -> individualized ICI plan.

Decision matrix:
                       Low CVD accel risk        High CVD accel risk
  High onco benefit    Standard ICI              Prefer dual-benefit drug
                                                  (or Standard ICI + cardiac protection)
  Low  onco benefit    Consider ICI              Avoid ICI
                                                  (or dual-benefit only)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preprocess import PRETRAINED_DIR

logger = logging.getLogger(__name__)

# Coarse per-cancer ORR priors (literature-derived).
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
    """Generate an individualized ICI recommendation by combining three signals.

    Example::

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

    # ── Main recommendation entry point ───────────────────────────────────────

    def generate_recommendation(
        self,
        patient_expr: dict[str, float],
        cancer_type: str,
        cvd_susceptibility: float = 0.5,
        candidate_drugs: list[str] | None = None,
    ) -> dict:
        """Generate an individualized ICI treatment recommendation.

        Parameters
        ----------
        patient_expr : dict[str, float]
            Tumor checkpoint expression profile (z-score).
        cancer_type : str
            Cancer code (e.g. "NSCLC", "SKCM").
        cvd_susceptibility : float
            Baseline CVD risk in [0, 1].
        candidate_drugs : list[str] | None
            Candidate ICI list; None considers every supported drug.

        Returns
        -------
        dict
            Keys: patient_cancer, cvd_susceptibility, ranked_drugs (DataFrame),
            primary_recommendation, recommendation (text).
        """
        if candidate_drugs is None:
            candidate_drugs = list(ONCO_PRIOR_ORR.keys())

        # Load the dual-benefit atlas (drug -> tier).
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

        # Pick the primary recommendation (highest priority row).
        primary = df.iloc[0] if not df.empty else None
        rec = self._generate_text(primary, df, cancer_type, cvd_susceptibility)

        return {
            "patient_cancer": cancer_type,
            "cvd_susceptibility": cvd_susceptibility,
            "ranked_drugs": df,
            "primary_recommendation": primary["drug"] if primary is not None else "None",
            "recommendation": rec,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

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
        """Recommendation priority: high benefit + low CVD acceleration ranks first."""
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
        risk_level = "high" if cvd_susc > 0.6 else ("moderate" if cvd_susc > 0.3 else "low")
        lines = [
            "== Individualized ICI Recommendation ==",
            f"Cancer type: {cancer_type} | Baseline CVD risk: {risk_level} ({cvd_susc:.2f})\n",
        ]

        if primary is None:
            lines.append("No eligible ICI candidate.")
            return "\n".join(lines)

        lines.append(f"Primary recommendation: {primary['drug']}")
        lines.append(f"  Tumor benefit (est.):   {primary['onco_benefit']:.0%}")
        lines.append(f"  CVD acceleration score: {primary['cvd_accel_score']:+.3f}")
        lines.append(f"  Net benefit:            {primary['net_benefit']:+.3f}")
        lines.append(f"  Category:               {primary['category']}")

        if cvd_susc > 0.6 and primary["cvd_accel_score"] > 0.3:
            lines.append("\n[!] Cardiac-protection suggestions:")
            lines.append("  - Co-administer statin +/- aspirin (plaque stabilization)")
            lines.append("  - Baseline echocardiogram, repeat every 3 months")
            lines.append("  - If CVD event risk is very high, consider switching to "
                         "magrolimab (dual-benefit)")

        dual_benefit = df[df["category"].str.contains("CVD-protective|dual-benefit",
                                                        case=False, na=False)]
        if not dual_benefit.empty:
            lines.append(
                f"\nDual-benefit alternatives: {', '.join(dual_benefit['drug'].tolist())}"
            )

        return "\n".join(lines)

    # ── Batch demo scenarios ──────────────────────────────────────────────────

    def demo_scenarios(self) -> pd.DataFrame:
        """Run three illustrative clinical scenarios."""
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
            logger.info("Scenario [%s]:\n%s", s["name"], rec["recommendation"])

        df = pd.DataFrame(rows)
        if self.out_dir is not None:
            df.to_parquet(self.out_dir / "demo_scenarios.parquet", index=False)
        return df
