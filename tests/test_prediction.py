"""Smoke tests for the CheckCardioNet prediction tool."""
from __future__ import annotations

import pandas as pd
import pytest

from checkcardionet.scoring import (
    ICIDrivenCVDAccelerationScore,
    IntegratedRecommendationSystem,
)
from checkcardionet.scoring.cvd_acceleration_score import ICI_CVD_PRIOR
from checkcardionet.data.preprocess import (
    PRETRAINED_DIR,
    load_checkpoint_panel,
)


def test_pretrained_artifacts_bundled():
    """All three pretrained parquet files must ship with the package."""
    for name in ("mr_results.parquet",
                 "bidirectional_scores.parquet",
                 "dual_benefit_atlas.parquet"):
        path = PRETRAINED_DIR / name
        assert path.exists(), f"Missing bundled artifact: {path}"


def test_checkpoint_panel_loads():
    panel = load_checkpoint_panel("all_checkpoints")
    assert len(panel) > 30
    assert "PDCD1" in panel
    assert "CD47" in panel


def test_supported_drugs_include_anchors():
    assert "pembrolizumab" in ICI_CVD_PRIOR
    assert "magrolimab"    in ICI_CVD_PRIOR
    assert ICI_CVD_PRIOR["magrolimab"]["cvd_risk_prior"] < 0  # CVD-protective


def test_single_patient_single_drug():
    scorer = ICIDrivenCVDAccelerationScore()
    result = scorer.compute(
        {"PDCD1": 2.5, "CD274": 1.8, "CD47": 0.5},
        proposed_ici="pembrolizumab",
        cvd_susceptibility=0.6,
    )
    assert -1.0 <= result["accel_score"] <= 1.0
    assert "PDCD1" in result["components"]
    assert isinstance(result["interpretation"], str) and result["interpretation"]


def test_recommendation_returns_primary():
    system = IntegratedRecommendationSystem()
    rec = system.generate_recommendation(
        patient_expr={"PDCD1": 2.0, "CD47": 0.5},
        cancer_type="NSCLC",
        cvd_susceptibility=0.2,
    )
    assert isinstance(rec["ranked_drugs"], pd.DataFrame)
    assert not rec["ranked_drugs"].empty
    assert rec["primary_recommendation"] in rec["ranked_drugs"]["drug"].tolist()


def test_high_cvd_protective_drug_ranks_higher():
    """For a high-CVD-risk patient with strong CD47 expression, magrolimab
    (CVD-protective) should outrank PD-1 monotherapy on net benefit."""
    system = IntegratedRecommendationSystem()
    rec = system.generate_recommendation(
        patient_expr={"PDCD1": 2.0, "CD47": 3.0, "SIRPA": 1.5},
        cancer_type="AML",
        cvd_susceptibility=0.8,
        candidate_drugs=["pembrolizumab", "magrolimab"],
    )
    df = rec["ranked_drugs"].set_index("drug")
    assert df.loc["magrolimab", "net_benefit"] >= df.loc["pembrolizumab", "net_benefit"]


@pytest.mark.parametrize("cvd", [0.0, 0.5, 1.0])
def test_recommendation_runs_at_cvd_extremes(cvd):
    system = IntegratedRecommendationSystem()
    rec = system.generate_recommendation(
        patient_expr={"PDCD1": 1.5},
        cancer_type="SKCM",
        cvd_susceptibility=cvd,
    )
    assert "recommendation" in rec and rec["recommendation"]
