"""Read-only exporter of completed R12 pilot outputs (no training)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def collect(repo: Path, output: Path, scenario: str, realization: int):
    result = repo / "results" / "revision" / "r12_native_sian_pairwise" / scenario / f"realization_{realization:02d}"
    if not result.is_dir():
        raise FileNotFoundError(f"Pilot folder not found: {result}")
    summary = result / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Successful pilot summary missing: {summary}")
    status = json.loads(summary.read_text(encoding="utf-8")).get("status")
    if status != "complete":
        raise RuntimeError(f"Pilot not complete: status={status!r}")
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    files = []
    for path in sorted(result.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        # No trained model is needed to validate reported numeric evidence;
        # include it only when modest in size, with a size-guard manifest.
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        files.append(path)
    costroot = repo / "results" / "revision" / "r12_cost_audit" / f"r12_{scenario.lower()}_{realization:02d}_sian"
    if costroot.exists():
        files += sorted(p for p in costroot.rglob("*") if p.is_file() and not p.is_symlink()
                        and p.stat().st_size <= 16 * 1024 * 1024)
    other_cost = repo / "results" / "revision" / "r12_agnam_canonical_cost" / scenario / f"realization_{realization:02d}"
    if other_cost.exists():
        files += sorted(p for p in other_cost.rglob("*") if p.is_file() and not p.is_symlink()
                        and p.stat().st_size <= 16 * 1024 * 1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "completed_pilot_evidence_only", "scenario": scenario,
                "realization": realization, "created_utc": datetime.now(timezone.utc).isoformat(),
                "files": {str(p.relative_to(repo)).replace("\\", "/"):
                          {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                          for p in files}}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=str(p.relative_to(repo)).replace("\\", "/"))
        z.writestr("R12_EVIDENCE_MANIFEST.json", json.dumps(manifest, indent=2))
    print("=== AG-NAM R12 SIAN PILOT EVIDENCE ===")
    print("FILES:", len(files), "| ARCHIVE:", output)
    print("TRAINING: NO | EXISTING OUTPUTS MODIFIED: NO")
    return output


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--scenario", default="S1", choices=["S1", "S2", "S3", "S4"])
    a.add_argument("--realization", type=int, default=1)
    a.add_argument("--output", default="AG_NAM_R12_SIAN_S1_Pilot_Evidence.zip")
    args = a.parse_args()
    collect(Path.cwd().resolve(), Path(args.output).resolve(), args.scenario, args.realization)


if __name__ == "__main__":
    main()
