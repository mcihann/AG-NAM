"""Run R8 exploratory calibration/resource audit without redoing discovery."""
from __future__ import annotations

import argparse
from agnam.revision.openml_calibration_runner import R8_TASK_IDS, run_r8_task
from agnam.utils.reproducibility import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--task-id', type=int)
    group.add_argument('--all', action='store_true')
    parser.add_argument('--force', action='store_true', help='Explicitly replace pre-existing R8 outputs')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    args = parser.parse_args()
    tasks = R8_TASK_IDS if args.all else (args.task_id,)
    device = get_device(args.device)
    print('R8 FINAL-FIT ONLY:', len(tasks), 'tasks /', 2*len(tasks), 'models', 'device', device)
    for pos, task_id in enumerate(tasks, 1):
        print(f'[{pos}/{len(tasks)}] TASK {task_id}')
        run_r8_task(task_id, device=device, force=args.force)
    print('R8 TASK EXECUTION COMPLETE')


if __name__ == '__main__':
    main()
