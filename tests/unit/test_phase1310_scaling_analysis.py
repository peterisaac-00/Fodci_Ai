from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_scaling_config_keeps_default_and_targets_approximately_50m() -> None:
    from backend_ai.model import FodciModel
    from scripts.analyze_phase1310_scaling import DEFAULT_CONFIG, SCALED_CONFIG

    default_model = FodciModel(DEFAULT_CONFIG)
    scaled_model = FodciModel(SCALED_CONFIG)
    assert default_model.num_parameters == 11_424_400
    assert 20_000_000 <= scaled_model.num_parameters <= 30_000_000
    assert scaled_model.num_parameters > default_model.num_parameters * 2
    assert SCALED_CONFIG.hidden_size % SCALED_CONFIG.num_attention_heads == 0


def test_phase1310_report_preserves_default_runtime_and_records_measurements() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase1310_scaling_analysis.json").read_text(encoding="utf-8"))
    default = report["models"]["default_11m"]
    scaled = report["models"]["scaled_candidate"]
    assert report["format"] == "fodci.phase1310_scaling_analysis"
    assert report["phase"] == "13.10"
    assert default["parameter_count"] == 11_424_400
    assert 20_000_000 <= scaled["parameter_count"] <= 30_000_000
    assert report["comparison"]["checkpoint_shape_compatible"] is False
    assert report["comparison"]["default_runtime_changed"] is False
    assert report["scaled_checkpoint_saved"] is True
    assert Path(report["scaled_checkpoint_path"]).is_file()
    assert report["comparison"]["decision"] == "retain_11m_default"
    assert report["torch_num_threads"] == 1
    assert default["forward_backward"]["gradients_finite"] is True
    assert scaled["forward_backward"]["gradients_finite"] is True
    assert default["short_training"]["finite_losses"] is True
    assert scaled["short_training"]["finite_losses"] is True
    assert scaled["short_training"]["global_step"] == 4
    assert scaled["short_training"]["parameters_changed"] is True
    assert report["validation_gates"]["all_passed"] is True
    assert report["validation_gates"]["scaled_checkpoint_reload"] is True
    assert report["validation_reload_delta"] < 1e-8


def test_phase1310_reports_resource_multiplier_and_saved_scaled_checkpoint() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase1310_scaling_analysis.json").read_text(encoding="utf-8"))
    default = report["models"]["default_11m"]
    scaled = report["models"]["scaled_candidate"]
    assert report["comparison"]["parameter_multiplier"] > 2.0
    assert scaled["forward_backward"]["elapsed_seconds"] > default["forward_backward"]["elapsed_seconds"]
    assert scaled["short_training"]["elapsed_seconds"] > default["short_training"]["elapsed_seconds"]
    assert scaled["forward_backward"]["rss_megabytes_after_backward"] > default["forward_backward"]["rss_megabytes_after_backward"]
    assert report["comparison"]["semantic_advantage_proven"] is False


def test_phase1310_markdown_explains_scaling_decision_and_evidence_limits() -> None:
    content = (ROOT / "docs" / "experiments" / "phase1310_scaling_analysis.md").read_text(encoding="utf-8")
    assert "Model Scaling Analysis" in content
    assert "approximately 26M" in content
    assert "retain the 11.4M model as the default" in content
    assert "not a fair capability comparison" in content
    assert "scaled checkpoint saved" in content.lower()
    assert "bounded four-step CPU training run" in content
