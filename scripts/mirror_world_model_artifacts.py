#!/usr/bin/env python3
"""Mirror small GOAL_RD world-model report artifacts for offline review."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("/mnt/cephfs_home_tianming.sha/grpo_alfworld/logs")
DEFAULT_OUTPUT_ROOT = Path("remote_docs/world_model")
DIAGNOSTIC_REPORT_FILES = (
    "checkpoint_diagnostics_report.md",
    "checkpoint_diagnostics_report.csv",
    "checkpoint_diagnostics_report.svg",
    "checkpoint_scores_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="gpudev logs root containing world_model_results.* and world_model_diagnostics/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Repository-local mirror destination.",
    )
    parser.add_argument(
        "--ssh-host",
        help="Optional SSH host, e.g. gpudev. When set, source-root is read remotely via ssh/scp.",
    )
    return parser.parse_args()


def markdown_metadata(markdown_path: Path) -> dict[str, str]:
    text = markdown_path.read_text(encoding="utf-8")
    metadata = {}
    for key in ("Branch", "Report revision"):
        match = re.search(r"^- " + re.escape(key) + r": `([^`]*)`", text, re.MULTILINE)
        metadata[key] = match.group(1) if match else ""
    return metadata


def results_counts(csv_path: Path) -> tuple[int, int]:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), sum(row.get("expected") == "yes" for row in rows)


def copy_file(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def copy_remote_file(ssh_host: str, source: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", f"{ssh_host}:{source}", str(destination)], check=True)
    return destination


def mirror_artifacts(source_root: Path, output_root: Path) -> list[Path]:
    copied = [
        copy_file(source_root / "world_model_results.md", output_root / "world_model_results.md"),
        copy_file(source_root / "world_model_results.csv", output_root / "world_model_results.csv"),
    ]

    diagnostics_root = source_root / "world_model_diagnostics"
    if diagnostics_root.is_dir():
        for diagnostic_dir in sorted(path for path in diagnostics_root.iterdir() if path.is_dir()):
            for name in DIAGNOSTIC_REPORT_FILES:
                source = diagnostic_dir / name
                if source.is_file():
                    copied.append(copy_file(source, output_root / diagnostics_root.name / diagnostic_dir.name / name))

    write_readme(source_root, output_root, copied)
    copied.append(output_root / "README.md")
    return copied


def remote_relative_path(source_root: Path, remote_path: str) -> Path:
    source_prefix = str(source_root).rstrip("/") + "/"
    if not remote_path.startswith(source_prefix):
        raise ValueError(f"{remote_path} is not under {source_root}")
    return Path(remote_path[len(source_prefix) :])


def remote_diagnostic_report_paths(ssh_host: str, source_root: Path) -> list[str]:
    diagnostics_root = source_root / "world_model_diagnostics"
    name_filter = " -o ".join(f"-name {shlex.quote(name)}" for name in DIAGNOSTIC_REPORT_FILES)
    command = (
        f"test -d {shlex.quote(str(diagnostics_root))} "
        f"&& find {shlex.quote(str(diagnostics_root))} -mindepth 2 -maxdepth 2 -type f \\( {name_filter} \\) "
        "|| true"
    )
    result = subprocess.run(
        ["ssh", ssh_host, command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def mirror_remote_artifacts(ssh_host: str, source_root: Path, output_root: Path) -> list[Path]:
    copied = [
        copy_remote_file(ssh_host, str(source_root / "world_model_results.md"), output_root / "world_model_results.md"),
        copy_remote_file(ssh_host, str(source_root / "world_model_results.csv"), output_root / "world_model_results.csv"),
    ]
    for remote_path in remote_diagnostic_report_paths(ssh_host, source_root):
        copied.append(copy_remote_file(ssh_host, remote_path, output_root / remote_relative_path(source_root, remote_path)))

    write_readme(source_root, output_root, copied)
    copied.append(output_root / "README.md")
    return copied


def write_readme(source_root: Path, output_root: Path, copied: list[Path]) -> None:
    metadata = markdown_metadata(output_root / "world_model_results.md")
    row_count, expected_count = results_counts(output_root / "world_model_results.csv")
    relative_files = sorted(path.relative_to(output_root) for path in copied)
    lines = [
        "# World-Model GOAL_RD Artifacts",
        "",
        "This directory mirrors the current small report artifacts from gpudev for offline review.",
        "",
        f"- Source root: `{source_root}`",
        f"- Branch: `{metadata.get('Branch') or '(unknown)'}`",
        f"- Report revision: `{metadata.get('Report revision') or '(unknown)'}`",
        f"- Rows: `{row_count}` total, `{expected_count}` expected GOAL_RD rows",
        "- Mirrored files:",
    ]
    lines.extend(f"  - `{path}`" for path in relative_files)
    lines.extend(
        [
            "",
            "The mirrored reports are snapshots. The authoritative live artifacts remain on gpudev under the source root above.",
            "",
            "Regenerate with `python scripts/mirror_world_model_artifacts.py` after refreshing the gpudev reports.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.ssh_host:
        copied = mirror_remote_artifacts(args.ssh_host, args.source_root, args.output_root)
    else:
        copied = mirror_artifacts(args.source_root, args.output_root)
    sys.stdout.write(f"MIRROR_DONE output={args.output_root} files={len(copied)}\n")


if __name__ == "__main__":
    main()
