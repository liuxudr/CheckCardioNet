"""Data utilities for the CheckCardioNet prediction tool.

A minimal helper module: locates packaged config / pretrained data and exposes
the immune-checkpoint gene panel.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configs and pretrained data ship inside the wheel — they resolve correctly
# whether the package is installed via pip / pip-from-git or run in editable mode.
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
PRETRAINED_DIR = Path(__file__).parent / "pretrained"


def load_checkpoint_panel(panel_key: str = "all_checkpoints") -> list[str]:
    """Load a checkpoint gene list from the bundled checkpoint_panel.yaml.

    Parameters
    ----------
    panel_key:
        e.g. "all_checkpoints", "adaptive_inhibitory", "innate_checkpoints",
        "inhibitory_ligands", ... (see configs/checkpoint_panel.yaml).
    """
    import yaml

    panel_file = CONFIGS_DIR / "checkpoint_panel.yaml"
    with open(panel_file) as f:
        panel = yaml.safe_load(f)
    if panel_key in panel:
        return list(panel[panel_key])
    # Fall back: when "all_checkpoints" key is missing, merge every list value.
    if panel_key == "all_checkpoints":
        merged: list[str] = []
        for v in panel.values():
            if isinstance(v, list):
                merged.extend(v)
        return sorted(set(merged))
    return []
