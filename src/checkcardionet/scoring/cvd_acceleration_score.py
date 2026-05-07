"""ICI-driven CVD acceleration risk score.

Formula:
  CVD_accel_score = sum over c in targets(ICI) of
      weight_c x MR_cvd_effect_c x patient_susceptibility_c

Components:
  weight_c                 - checkpoint expression in the patient's tumor
                              (z-score / log-TPM, from eQTL or RNA-seq).
  MR_cvd_effect_c          - Mendelian-randomization causal effect of the
                              checkpoint on CVD (IVW beta).
  patient_susceptibility_c - baseline CVD susceptibility x checkpoint
                              bidirectionality (BDS).

High score => ICI regimen is likely to significantly accelerate CVD in this patient.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preprocess import PRETRAINED_DIR, load_checkpoint_panel

logger = logging.getLogger(__name__)

# Per-drug targets and literature-derived CVD-risk priors.
# A negative prior (e.g. magrolimab) indicates a putative CVD-protective drug.
ICI_CVD_PRIOR: dict[str, dict] = {
    "pembrolizumab": {"targets": ["PDCD1"], "cvd_risk_prior": 0.7},
    "nivolumab":     {"targets": ["PDCD1"], "cvd_risk_prior": 0.7},
    "ipilimumab":    {"targets": ["CTLA4"], "cvd_risk_prior": 0.6},
    "atezolizumab":  {"targets": ["CD274"], "cvd_risk_prior": 0.5},
    "durvalumab":    {"targets": ["CD274"], "cvd_risk_prior": 0.5},
    "relatlimab":    {"targets": ["LAG3"],  "cvd_risk_prior": 0.3},
    "tiragolumab":   {"targets": ["TIGIT"], "cvd_risk_prior": 0.3},
    "cobolimab":     {"targets": ["HAVCR2"],"cvd_risk_prior": 0.5},
    "magrolimab":    {"targets": ["CD47"],  "cvd_risk_prior": -0.3},  # CVD-protective
    "monalizumab":   {"targets": ["KLRC1"], "cvd_risk_prior": 0.2},
    "MK-4830":       {"targets": ["LILRB2"],"cvd_risk_prior": 0.2},
}


class ICIDrivenCVDAccelerationScore:
    """Estimate the per-patient risk that an ICI regimen accelerates CVD progression.

    Example::

        scorer = ICIDrivenCVDAccelerationScore()
        # Patient checkpoint expression profile (z-score / log-TPM)
        patient_expr = {"PDCD1": 2.5, "CD274": 1.8, "CD47": 3.1}
        result = scorer.compute(patient_expr,
                                proposed_ici="pembrolizumab",
                                cvd_susceptibility=0.6)
    """

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._mr_effects = self._load_mr_effects()
        self._bds = self._load_bds()

    def _load_mr_effects(self) -> dict[str, float]:
        """Load median per-gene MR causal effects on CVD (IVW beta)."""
        path = PRETRAINED_DIR / "mr_results.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return df.groupby("gene")["ivw_beta"].median().to_dict()

    def _load_bds(self) -> dict[str, float]:
        """Load per-gene bidirectionality scores (BDS)."""
        path = PRETRAINED_DIR / "bidirectional_scores.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return df["BDS"].to_dict()

    # ── Single-patient scoring ────────────────────────────────────────────────

    def compute(
        self,
        patient_expr: dict[str, float],
        proposed_ici: str,
        cvd_susceptibility: float = 0.5,
    ) -> dict:
        """Compute the ICI-CVD acceleration score for a single patient.

        Parameters
        ----------
        patient_expr : dict[str, float]
            Tumor checkpoint expression profile (z-score or log-TPM).
        proposed_ici : str
            Name of the candidate ICI drug.
        cvd_susceptibility : float
            Baseline CVD susceptibility in [0, 1].

        Returns
        -------
        dict
            Keys: drug, targets, accel_score, cvd_susceptibility,
            components (per-target decomposition), interpretation.
        """
        ici_info = ICI_CVD_PRIOR.get(proposed_ici, {})
        targets = ici_info.get("targets", [])
        cvd_risk_prior = ici_info.get("cvd_risk_prior", 0.3)

        components = {}
        for target in targets:
            # Patient expression weight, normalized to [0, 1].
            expr = patient_expr.get(target, 0.0)
            expr_weight = float(np.clip((expr + 3) / 6, 0, 1))  # assumes z-score in [-3, +3]

            # MR effect (data-driven; fall back to drug-level prior if missing).
            mr_beta = self._mr_effects.get(target, cvd_risk_prior * 0.5)

            # Bidirectionality modifier.
            bds = self._bds.get(target, 0.0)
            bidirectionality_mod = 1.0 + abs(bds) * 0.3

            comp = expr_weight * abs(mr_beta) * bidirectionality_mod * cvd_susceptibility
            if mr_beta < 0:
                comp = -comp  # CVD-protective target
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
            return (f"HIGH risk: {drug} may significantly accelerate CVD progression. "
                    f"Consider an alternative regimen or aggressive cardiac protection.")
        elif score > 0.2:
            return (f"MODERATE risk: {drug} may moderately accelerate CVD. "
                    f"Close cardiovascular monitoring is recommended.")
        elif score > 0:
            return (f"LOW risk: {drug} has limited impact on CVD; standard cardiac "
                    f"monitoring is sufficient.")
        else:
            return f"CVD-PROTECTIVE: {drug} may confer cardiovascular protection."

    # ── Batch scoring (multi-patient x multi-drug) ────────────────────────────

    def batch_score(
        self,
        patient_profiles: pd.DataFrame,
        ici_drugs: list[str] | None = None,
        expr_cols: list[str] | None = None,
        cvd_col: str = "cvd_susceptibility",
    ) -> pd.DataFrame:
        """Score multiple patients across multiple candidate ICIs.

        Parameters
        ----------
        patient_profiles : pd.DataFrame
            One row per patient. Must contain expression columns and a
            cvd_susceptibility column.
        ici_drugs : list[str] | None
            Candidate ICI list; None uses every supported drug.
        expr_cols : list[str] | None
            Expression column names (gene symbols); None auto-detects from the
            built-in checkpoint panel.

        Returns
        -------
        pd.DataFrame
            Rows = patients, columns = '<drug>_accel'.
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

    # ── Demo cases ────────────────────────────────────────────────────────────

    def demo_case_studies(self) -> pd.DataFrame:
        """Run three illustrative cases."""
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
                "[%s] drug=%s accel=%.3f -- %s",
                r["case"], r["proposed_ici"], r["accel_score"], r["interpretation"],
            )
        return df
