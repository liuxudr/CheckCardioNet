"""Data utilities for the CheckCardioNet prediction tool.

仅保留预测工具所需的最小函数集：配置/预训练数据加载、检查点基因清单。
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 配置文件 & 预训练数据均打包到 wheel 内部，pip/pip-from-git 安装即可使用
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
PRETRAINED_DIR = Path(__file__).parent / "pretrained"


def load_checkpoint_panel(panel_key: str = "all_checkpoints") -> list[str]:
    """从打包的 checkpoint_panel.yaml 读取检查点基因列表。

    Parameters
    ----------
    panel_key:
        e.g. "all_checkpoints", "adaptive_inhibitory", "innate_checkpoints",
        "inhibitory_ligands", … (see configs/checkpoint_panel.yaml)
    """
    import yaml

    panel_file = CONFIGS_DIR / "checkpoint_panel.yaml"
    with open(panel_file) as f:
        panel = yaml.safe_load(f)
    if panel_key in panel:
        return list(panel[panel_key])
    # 兼容：组合若干分类
    if panel_key == "all_checkpoints":
        merged: list[str] = []
        for v in panel.values():
            if isinstance(v, list):
                merged.extend(v)
        return sorted(set(merged))
    return []
