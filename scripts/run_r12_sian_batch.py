"""Safely launch checkpointed R12 SIAN-native pairwise runs using the locked pilot.

Dry-run by default. The script never modifies R1-R11 artifacts and never
restarts a completed or partially written realization automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

EXPECTED_PILOT_SHA256 = "41166131baac2c6b11995f1639479781f16521892ad52e3cb5345849c70a1d47"
RESULT_ROOT = Path("results/revision/r12_native_sian_pairwise")
COST_ROOT = Path("results/revision/r12_cost_audit")


def scenario_range(scenario: str, first: int, last: int) -> tuple[tuple[str, int], ...]:
    if scenario not in ("S1", "S2", "S3", "S4"):
        raise ValueError("Scenario must be S1, S2, S3 or S4")
    if not (1 <= first <= last <= 20):
        raise ValueError("Realization interval must satisfy 1 <= first <= last <= 20")
    return tuple((scenario, number) for number in range(first, last + 1))


def run_paths(repo: Path, scenario: str, number: int) -> tuple[Path, Path, Path]:
    label = f"r12_{scenario.lower()}_{number:02d}_sian"
    result = repo / RESULT_ROOT / scenario / f"realization_{number:02d}"
    cost = repo / COST_ROOT / label
    return result, Path(str(result) + ".inprogress"), cost


def validate_locked_source(repo: Path) -> str:
    pilot = repo / "scripts/run_r12_sian_pairwise_pilot.py"
    profiler = repo / "scripts/r12_profile_command.py"
    if not pilot.is_file() or not profiler.is_file():
        raise FileNotFoundError("Missing existing R12 pilot or cost-profiler script")
    actual = hashlib.sha256(pilot.read_bytes()).hexdigest()
    if actual != EXPECTED_PILOT_SHA256:
        raise RuntimeError("Pilot source has changed since audited S1-01. Refuse mixed protocols. "
                           f"Current SHA256={actual}; expected={EXPECTED_PILOT_SHA256}")
    return actual


def classify(repo: Path, scenario: str, number: int) -> str:
    result, staging, cost = run_paths(repo, scenario, number)
    if staging.exists():
        return "inprogress_or_failed"
    if result.exists() and cost.exists():
        summary_path = result / "summary.json"
        measurement_path = cost / "measurement.json"
        if summary_path.is_file() and measurement_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            meas = json.loads(measurement_path.read_text(encoding="utf-8"))
            if (summary.get("status") == "complete" and summary.get("scenario") == scenario
                and summary.get("realization") == number and
                meas.get("experiment_run_status") == "complete" and meas.get("returncode") == 0):
                for f in ("selected_pairs.csv", "test_predictions.csv", "sian_state_dict.pt"):
                    if not (result / f).is_file():
                        return "partial"
                return "complete"
        return "partial"
    if result.exists() or cost.exists():
        return "partial"
    return "new"


def command(scenario: str, number: int) -> list[str]:
    label = f"r12_{scenario.lower()}_{number:02d}_sian"
    return [sys.executable, "scripts/r12_profile_command.py", "--label", label,
            "--execute", "--", sys.executable,
            "scripts/run_r12_sian_pairwise_pilot.py", "--scenario", scenario,
            "--realization", str(number), "--execute"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("S1", "S2", "S3", "S4"), default="S1")
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--execute", action="store_true", help="Actually train missing realizations")
    args = parser.parse_args(argv)
    repo = Path.cwd().resolve()
    validate_locked_source(repo)
    tasks = scenario_range(args.scenario, args.start, args.end)
    print("=== R12 CHECKPOINTED NATIVE SIAN BATCH ===", flush=True)
    print("SCENARIO:", args.scenario, "RANGE:", args.start, "-", args.end, flush=True)
    print("PILOT SHA256:", EXPECTED_PILOT_SHA256, flush=True)
    print("TRAINING:", "YES" if args.execute else "NO (DRY RUN)", flush=True)
    statuses = [(s, n, classify(repo, s, n)) for s, n in tasks]
    for s, n, status in statuses:
        print(f"{s}-{n:02d} {status}", flush=True)
    blockers = [(s, n, status) for s, n, status in statuses if status not in ("complete", "new")]
    if blockers:
        print("BLOCKED: Existing partial/inprogress run(s) must be examined manually:", blockers)
        return 2
    if not args.execute:
        print("DRY RUN COMPLETE; no process launched or output modified.")
        return 0
    for s, n, status in statuses:
        if status == "complete":
            print(f"SKIP {s}-{n:02d}: existing completed output retained", flush=True)
            continue
        cmd = command(s, n)
        print(f"START {s}-{n:02d}: profiler + SIAN training", flush=True)
        try:
            completed = subprocess.run(cmd, cwd=repo, check=False)
        except KeyboardInterrupt:
            print("INTERRUPTED: no deletion or restart was performed", flush=True)
            return 130
        if completed.returncode != 0:
            print(f"STOP {s}-{n:02d}: exit={completed.returncode}; inspect corresponding cost log", flush=True)
            return completed.returncode or 1
        after = classify(repo, s, n)
        if after != "complete":
            print(f"STOP {s}-{n:02d}: unexpected post-run status {after}", flush=True)
            return 3
        print(f"COMPLETE {s}-{n:02d}; checkpoint saved", flush=True)
    print("BATCH COMPLETE: all requested runs validated at status level", flush=True)
    print("NEXT: use scripts/collect_r12_sian_batch.py for numerical evidence audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
