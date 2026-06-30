import csv
import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "mirror_world_model_artifacts.py"
    spec = importlib.util.spec_from_file_location("mirror_world_model_artifacts_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mirror_artifacts_copies_small_reports_and_writes_manifest(tmp_path):
    module = _load_module()
    source_root = tmp_path / "logs"
    diagnostics_dir = source_root / "world_model_diagnostics" / "diag_a"
    diagnostics_dir.mkdir(parents=True)
    (source_root / "world_model_results.md").write_text(
        "\n".join(
            [
                "# World Model Results",
                "- Branch: `world-model-latent-objective`",
                "- Report revision: `abc123`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with (source_root / "world_model_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_key", "expected"])
        writer.writeheader()
        writer.writerows(
            [
                {"run_key": "grpo_baseline_s0", "expected": "yes"},
                {"run_key": "extra", "expected": "no"},
            ]
        )
    for name in module.DIAGNOSTIC_REPORT_FILES:
        (diagnostics_dir / name).write_text(f"{name}\n", encoding="utf-8")
    (diagnostics_dir / "checkpoint_scores.csv").write_text("too large for mirror\n", encoding="utf-8")
    output_root = tmp_path / "mirror"

    copied = module.mirror_artifacts(source_root, output_root)

    copied_names = {path.relative_to(output_root).as_posix() for path in copied}
    assert "world_model_results.md" in copied_names
    assert "world_model_results.csv" in copied_names
    assert "world_model_diagnostics/diag_a/checkpoint_diagnostics_report.md" in copied_names
    assert "world_model_diagnostics/diag_a/checkpoint_diagnostics_report.csv" in copied_names
    assert "world_model_diagnostics/diag_a/checkpoint_diagnostics_report.svg" in copied_names
    assert "world_model_diagnostics/diag_a/checkpoint_scores_summary.json" in copied_names
    assert "world_model_diagnostics/diag_a/checkpoint_scores.csv" not in copied_names

    readme = (output_root / "README.md").read_text(encoding="utf-8")
    assert "- Branch: `world-model-latent-objective`" in readme
    assert "- Report revision: `abc123`" in readme
    assert "- Rows: `2` total, `1` expected GOAL_RD rows" in readme


def test_remote_relative_path_requires_source_root_prefix():
    module = _load_module()
    source_root = Path("/mnt/work/logs")

    assert module.remote_relative_path(
        source_root,
        "/mnt/work/logs/world_model_diagnostics/diag_a/checkpoint_diagnostics_report.csv",
    ) == Path("world_model_diagnostics/diag_a/checkpoint_diagnostics_report.csv")

    try:
        module.remote_relative_path(source_root, "/other/logs/world_model_results.csv")
    except ValueError as exc:
        assert "is not under" in str(exc)
    else:
        raise AssertionError("remote_relative_path should reject paths outside source_root")
