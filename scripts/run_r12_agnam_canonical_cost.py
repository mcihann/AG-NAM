"""Optional R12 AG-NAM cost audit of the *canonical six-model* benchmark.

WARNING: The canonical synthetic runner executes Main NAM, Full AG-NAM,
No-ISR, Single-Run, Random, and Oracle models. Its time is NOT the
isolated Full AG-NAM pipeline cost. It is an all-variants cost benchmark.
No stored experimental result is overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from agnam.benchmarking.synthetic_protocol import SyntheticBenchmarkProtocol, build_synthetic_schedule
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig, run_single_synthetic_benchmark,
)
from agnam.utils.reproducibility import get_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("S1", "S2", "S3", "S4"), default="S1")
    parser.add_argument("--realization", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", default="results/revision/r12_agnam_canonical_cost")
    a = parser.parse_args(argv)
    spec = next((x for x in build_synthetic_schedule(SyntheticBenchmarkProtocol())
                 if x.scenario == a.scenario and x.realization_number == a.realization), None)
    if spec is None:
        parser.error("Unknown realization")
    print("CANONICAL BENCHMARK (six variants):", spec.scenario, spec.realization_number)
    if not a.execute:
        print("CHECK ONLY: no training. Add --execute to start all six models.")
        return 0
    dst = Path(a.output_root) / spec.scenario / f"realization_{a.realization:02d}"
    if dst.exists():
        raise FileExistsError(f"Refusing overwrite: {dst}")
    dst.mkdir(parents=True)
    start = time.perf_counter()
    try:
        result = run_single_synthetic_benchmark(
            spec=spec, protocol=SyntheticBenchmarkProtocol(),
            model_config=SyntheticModelConfig(), device=get_device(), verbose=True)
        elapsed = time.perf_counter() - start
        (dst / "summary.json").write_text(json.dumps({
            "status": "complete", "scope": "canonical all-six-model synthetic realization",
            "not_comparable_to_sian_runtime": True,
            "wall_clock_seconds": elapsed, "record": asdict(result.record)
        }, indent=2, default=str), encoding="utf-8")
        print("CANONICAL BENCHMARK COMPLETE, all variants (s):", elapsed)
        return 0
    except BaseException as e:
        (dst / "failed.json").write_text(json.dumps({"status": "failed", "error": repr(e)}, indent=2),
                                         encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
