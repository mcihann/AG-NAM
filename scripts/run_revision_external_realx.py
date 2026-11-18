from __future__ import annotations
import argparse
from agnam.revision.external_realx_runner import select_specs, run_external_realx


def main():
    p = argparse.ArgumentParser(description="Run R6 Real-X external recovery")
    p.add_argument("--strength", choices=["moderate", "null", "both"], default="moderate")
    p.add_argument("--task-id", type=int)
    p.add_argument("--realization", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Replace selected results only after documented failure")
    a = p.parse_args()
    specs = select_specs(strength=a.strength, task_id=a.task_id, realization=a.realization)
    print("PLANNED RUNS:", len(specs), "METHOD EVALUATIONS:", 3*len(specs))
    for i,s in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {s.run_id}", flush=True)
        if not a.dry_run:
            run_external_realx(s, force=a.force)


if __name__ == "__main__":
    main()
