"""Small versioned registry for reproducible QXTI validation results."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "validation" / "results"
REGISTRY_PATH = RESULTS_DIR / "registry.json"
REPORT_PATH = PROJECT_ROOT / "validation" / "VALIDATION_REPORT.md"


def source_state() -> dict[str, Any]:
    """Return the Git revision and whether the benchmark used local changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", True
    return {"qxti_commit": commit, "working_tree_dirty": dirty}


def record_result(record: dict[str, Any]) -> None:
    """Insert or replace one result by stable ``id`` and rebuild the report."""
    if not record.get("id"):
        raise ValueError("A validation record requires a stable non-empty 'id'.")
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": 1, "records": []}

    normalized = dict(record)
    normalized.setdefault("date", date.today().isoformat())
    if normalized.get("artifact"):
        artifact = Path(normalized["artifact"])
        try:
            normalized["artifact"] = str(artifact.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            normalized["artifact"] = str(artifact)
    normalized.update(source_state())
    records = [item for item in registry.get("records", []) if item.get("id") != record["id"]]
    records.append(normalized)
    registry["records"] = sorted(records, key=lambda item: str(item.get("id", "")))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(registry), encoding="utf-8")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def render_report(registry: dict[str, Any]) -> str:
    records = list(registry.get("records", []))
    passed = sum(bool(item.get("passed")) for item in records)
    lines = [
        "# QXTI Validation Report",
        "",
        "This report is generated from `validation/results/registry.json`. "
        "A passing entry applies only to the scope stated in that entry; it does "
        "not validate unrelated QXTI observables.",
        "",
        "Each entry records how the benchmark was implemented, the provenance and "
        "independence level of its reference, the error calculation, and any "
        "production-code change required by the check. A separately coded formula "
        "is an independent reference, but it is not labeled as external software.",
        "",
        f"Recorded validations: **{len(records)}** · Passed: **{passed}** · "
        f"Failed: **{len(records) - passed}**",
        "",
        "## Validation workflow",
        "",
        "- Each `validation/benchmark_*.py` module is an executable, deterministic "
        "benchmark that builds QXTI and its reference through separate code paths.",
        "- Every run writes a raw JSON artifact, replaces its stable entry in "
        "`validation/results/registry.json`, and regenerates this report.",
        "- `tests/test_external_validation.py` invokes the same benchmark functions "
        "so scientific checks fail in the normal pytest regression suite.",
        "- The `validation` optional dependency group pins the external PythTB "
        "dependency used by the cross-code cases; closed-form and reconstruction "
        "references are explicitly classified as internal independent evidence.",
        "",
    ]
    for record in records:
        status = "PASS" if record.get("passed") else "FAIL"
        lines.extend(
            [
                f"## {record.get('title', record['id'])}",
                "",
                f"- Status: **{status}**",
                f"- Date: `{record.get('date', 'unknown')}`",
                f"- Scope: {record.get('scope', '')}",
                f"- Reference: {record.get('independent_reference', record.get('external_reference', ''))}",
                f"- Evidence type: {record.get('reference_type', 'not classified')}",
                f"- QXTI commit: `{record.get('qxti_commit', 'unknown')}`",
                f"- Working tree dirty: `{record.get('working_tree_dirty', True)}`",
                "",
            ]
        )
        rows = list(record.get("results", []))
        if rows:
            headers = list(rows[0])
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(_format_value(row.get(key, "")) for key in headers) + " |")
            lines.append("")
        implementation = list(record.get("implementation", []))
        if implementation:
            lines.extend(["Implementation:", ""])
            lines.extend(f"- {item}" for item in implementation)
            lines.append("")
        provenance = list(record.get("reference_provenance", []))
        if provenance:
            lines.extend(["Reference provenance:", ""])
            lines.extend(f"- {item}" for item in provenance)
            lines.append("")
        production_changes = list(record.get("production_code_changes", []))
        if production_changes:
            lines.extend(["Production-code changes motivated by this benchmark:", ""])
            lines.extend(f"- {change}" for change in production_changes)
            lines.append("")
        error_methodology = list(record.get("error_methodology", []))
        if error_methodology:
            lines.extend(["Error calculation:", ""])
            lines.extend(f"- {method}" for method in error_methodology)
            lines.append("")
        criteria = list(record.get("acceptance_criteria", []))
        if criteria:
            lines.extend(["Acceptance criteria:", ""])
            lines.extend(f"- {criterion}" for criterion in criteria)
            lines.append("")
        if record.get("conclusion"):
            lines.extend([f"Conclusion: {record['conclusion']}", ""])
        limitations = list(record.get("limitations", []))
        if limitations:
            lines.extend(["Limitations:", ""])
            lines.extend(f"- {limitation}" for limitation in limitations)
            lines.append("")
        if record.get("artifact"):
            artifact = Path(record["artifact"])
            try:
                artifact = artifact.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
            lines.extend([f"Raw artifact: `{artifact}`", ""])
    return "\n".join(lines).rstrip() + "\n"
