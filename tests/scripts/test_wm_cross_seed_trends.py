import csv
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "wm_cross_seed_trends.py"
    spec = importlib.util.spec_from_file_location(
        "wm_cross_seed_trends_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_trends(path, module, *, omit=None, paired_games=3553, sign=1.0):
    rows = []
    for level in module.LEVELS:
        for metric in module.METRICS:
            for statistic in module.STATISTICS:
                key = (level, metric, statistic)
                if key == omit:
                    continue
                slope = 0.01 * sign
                rows.append(
                    {
                        "level": level,
                        "metric": metric,
                        "statistic": statistic,
                        "paired_games": paired_games,
                        "slope_per_step": slope,
                        "slope_ci_lo": slope - 0.004,
                        "slope_ci_hi": slope + 0.004,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cross_seed_trends_confirm_same_significant_direction(tmp_path):
    module = _load_module()
    seed0_path = tmp_path / "seed0.csv"
    seed1_path = tmp_path / "seed1.csv"
    _write_trends(seed0_path, module)
    _write_trends(seed1_path, module)

    rows = module.compare_trends(
        module.load_trends(seed0_path),
        module.load_trends(seed1_path),
    )

    assert len(rows) == 40
    assert all(row["direction_consistent"] for row in rows)
    assert all(row["confirmed_by_second_seed"] for row in rows)


def test_cross_seed_trends_fail_closed_on_missing_rows_or_games(tmp_path):
    module = _load_module()
    missing_path = tmp_path / "missing.csv"
    _write_trends(
        missing_path,
        module,
        omit=("episode", "ce", "gap"),
    )
    with pytest.raises(ValueError, match="matrix is incomplete"):
        module.load_trends(missing_path)

    small_path = tmp_path / "small.csv"
    _write_trends(small_path, module, paired_games=2048)
    with pytest.raises(ValueError, match="expected exactly 3553"):
        module.load_trends(small_path)


def test_cross_seed_confirmation_requires_ci_and_point_same_side(tmp_path):
    module = _load_module()
    seed0_path = tmp_path / "seed0.csv"
    seed1_path = tmp_path / "seed1.csv"
    _write_trends(seed0_path, module)
    _write_trends(seed1_path, module)
    seed0 = module.load_trends(seed0_path)
    seed1 = module.load_trends(seed1_path)
    key = ("episode", "ce", "gap")
    for values in (seed0[key], seed1[key]):
        values["slope"] = 0.01
        values["ci_lo"] = -0.02
        values["ci_hi"] = -0.01

    row = next(item for item in module.compare_trends(seed0, seed1) if (item["level"], item["metric"], item["statistic"]) == key)

    assert row["direction_consistent"] is True
    assert row["both_cis_exclude_zero"] is True
    assert row["confirmed_by_second_seed"] is False
