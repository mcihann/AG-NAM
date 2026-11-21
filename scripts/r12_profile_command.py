#!/usr/bin/env python3
"""R12 full-command wall-clock / process-memory profiler.

This script profiles a *new explicit invocation*, NOT stored historical timing.
It never starts any training without --execute, never modifies code/results on
its own, and stores measurements separately in results/revision/r12_cost_audit.

Example (only after we verify an isolated-output pilot command):
  python scripts/r12_profile_command.py --label pilot_s1 --execute -- python ...

The command after '--' is executed verbatim. The operator must ensure its
arguments do not overwrite frozen experiment outputs. Default: dry run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


def git_info(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, args in [('head', ['rev-parse', 'HEAD']),
                      ('branch', ['branch', '--show-current']),
                      ('dirty', ['status', '--porcelain'])]:
        try:
            p = subprocess.run(['git'] + args, cwd=root, text=True,
                               capture_output=True, timeout=5, check=False)
            result[key] = p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    return result


def gpu_metrics() -> dict[int, float]:
    """Best-effort nvidia-smi GPU usage by process; optional/unsupported on some platforms."""
    if shutil.which('nvidia-smi') is None:
        return {}
    try:
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=3, check=False)
        if p.returncode:
            return {}
        usage: dict[int, float] = {}
        for line in p.stdout.splitlines():
            items = [x.strip() for x in line.split(',')]
            if len(items) < 2:
                continue
            try:
                usage[int(items[0])] = float(items[1])
            except ValueError:
                pass
        return usage
    except (OSError, subprocess.TimeoutExpired):
        return {}


def monitor(pid: int, stop: threading.Event, state: dict[str, Any], sample_s: float) -> None:
    try:
        import psutil
    except ImportError:
        state['memory_status'] = 'psutil unavailable; pip install psutil for RSS sampling'
        return
    state['memory_status'] = 'sampled process tree RSS; not exact continuous peak'
    tracked = 0
    while not stop.is_set():
        try:
            parent = psutil.Process(pid)
            processes = [parent] + parent.children(recursive=True)
            rss = 0
            live_pids = set()
            for proc in processes:
                try:
                    rss += proc.memory_info().rss
                    live_pids.add(proc.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            if rss:
                state['peak_rss_mib'] = max(state['peak_rss_mib'], rss / 1048576)
                state['rss_samples'] += 1
            state['max_process_count'] = max(state['max_process_count'], len(live_pids))
            # GPU figures are device-reported per PID, NOT CUDA peak allocated.
            gpu_mem = gpu_metrics()
            if gpu_mem:
                gpu_sum = sum(mib for p, mib in gpu_mem.items() if p in live_pids)
                state['peak_gpu_process_memory_mib'] = max(state['peak_gpu_process_memory_mib'], gpu_sum)
                state['gpu_samples'] += 1
            tracked += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        stop.wait(sample_s)
    state['poll_iterations'] = tracked


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='pilot', help='Short unique measurement identifier')
    ap.add_argument('--execute', action='store_true', help='Explicitly allow running the supplied command')
    ap.add_argument('--output-root', default='results/revision/r12_cost_audit',
                    help='Dedicated measurement directory')
    ap.add_argument('--sample-interval', type=float, default=0.5,
                    help='Process/GPU polling interval in seconds')
    ap.add_argument('command', nargs=argparse.REMAINDER,
                    help='Use -- followed by a command, e.g. -- python -m ...')
    args = ap.parse_args(argv)
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.label.replace('_', '').replace('-', '').isalnum():
        ap.error('label must contain only letters, numbers, underscore, hyphen')
    if not 0.1 <= args.sample_interval <= 10:
        ap.error('--sample-interval must be between 0.1 and 10 seconds')
    return args


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path.cwd().resolve()
    print('=== AG-NAM R12 END-TO-END COMMAND PROFILER ===')
    print('LABEL:', args.label)
    print('COMMAND:', args.command)
    if not args.execute:
        print('DRY RUN: --execute was not supplied; no process launched.')
        return 0
    if not args.command:
        raise SystemExit('ERROR: provide a command after -- when using --execute')
    dest = Path(args.output_root).resolve() / args.label
    # prevent accidental overwrites in repeat experiments
    if dest.exists():
        raise SystemExit(f'ERROR: measurement directory exists: {dest} (choose a new label)')
    dest.mkdir(parents=True)
    log = dest / 'command_output.log'
    state = {'peak_rss_mib': 0.0, 'rss_samples': 0,
             'peak_gpu_process_memory_mib': 0.0, 'gpu_samples': 0,
             'max_process_count': 0, 'memory_status': 'not sampled', 'poll_iterations': 0}
    # Do not shell=True. The command is passed exactly as argv.
    begun = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        with log.open('w', encoding='utf-8', errors='replace') as fout:
            p = subprocess.Popen(args.command, cwd=repo, stdout=fout,
                                 stderr=subprocess.STDOUT, shell=False)
            stop = threading.Event()
            thread = threading.Thread(target=monitor, args=(p.pid, stop, state, args.sample_interval), daemon=True)
            thread.start()
            try:
                return_code = p.wait()
            except KeyboardInterrupt:
                p.terminate()
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                return_code = 130
            finally:
                stop.set()
                thread.join(timeout=5)
    except OSError as e:
        return_code = 127
        log.write_text('PROCESS START FAILED: ' + repr(e), encoding='utf-8')
    elapsed = time.perf_counter() - start
    ended = datetime.now(timezone.utc).isoformat()
    data = {
        'analysis_name': 'r12_full_command_wallclock_cost_audit',
        'label': args.label, 'command': args.command,
        'experiment_run_status': 'complete' if return_code == 0 else 'failed',
        'returncode': return_code, 'elapsed_wall_seconds': elapsed,
        'begun_utc': begun, 'ended_utc': ended, 'sample_interval_seconds': args.sample_interval,
        'peak_process_tree_rss_mib_sampled': state['peak_rss_mib'] if state['rss_samples'] else None,
        'rss_samples': state['rss_samples'],
        'peak_nvidia_smi_process_gpu_memory_mib_sampled': (
            state['peak_gpu_process_memory_mib'] if state['gpu_samples'] else None),
        'gpu_samples': state['gpu_samples'],
        'max_process_count_sampled': state['max_process_count'],
        'memory_status': state['memory_status'],
        'timing_scope': 'whole launched command; includes only stages executed by that command',
        'comparability_cautions': [
            'CPU/GPU memory is sampled (not a true instantaneous peak).',
            'GPU process memory from nvidia-smi is not torch.cuda.max_memory_allocated().',
            'Existing cached/skipped stages are NOT counted if the command resumes rather than recomputes.',
            'Measured times are platform, command, implementation and cache dependent.',
            'No completed benchmark result is implied by a dry-run or by the profiler alone.'
        ],
        'platform': platform.platform(), 'python': sys.version.split()[0],
        'repo': str(repo), 'git': git_info(repo),
        'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    (dest / 'measurement.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('OUTPUT:', dest)
    print('EXIT CODE:', return_code)
    print('WALL TIME (s):', round(elapsed, 3))
    print('PEAK RSS (MiB, sampled):', data['peak_process_tree_rss_mib_sampled'])
    print('GPU PROCESS MEMORY (MiB, sampled):', data['peak_nvidia_smi_process_gpu_memory_mib_sampled'])
    print('NOTE: This is only a valid full AG-NAM cost if the supplied command ran the entire pipeline from scratch.')
    return return_code


if __name__ == '__main__':
    raise SystemExit(run())
