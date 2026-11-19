"""R10: audit domain case inventory or refit final AG-NAM for model evidence."""
from __future__ import annotations

import argparse

from agnam.revision.domain_grounded_validation import (
    LOCKED_CASE_TASK_IDS, R10_TASK_IDS, export_pair_inventory, run_r10_task,
)
from agnam.utils.reproducibility import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-only', action='store_true', help='No network access and no training')
    group.add_argument('--pilot', action='store_true', help='Climate-model locked task 146819')
    group.add_argument('--task-id', type=int, choices=R10_TASK_IDS)
    group.add_argument('--all-locked', action='store_true', help='All three pre-locked case studies')
    parser.add_argument('--device', choices=('cpu','cuda'), default=None)
    args = parser.parse_args()
    export_pair_inventory()
    if args.audit_only:
        return
    tasks = (146819,) if args.pilot else (args.task_id,) if args.task_id else LOCKED_CASE_TASK_IDS
    device = get_device(args.device)
    print('R10 FINAL-ONLY MODEL EVIDENCE:', tasks, 'device=', device)
    for task_id in tasks:
        run_r10_task(task_id, device=device)


if __name__ == '__main__':
    main()
