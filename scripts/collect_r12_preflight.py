#!/usr/bin/env python3
"""R12: read-only AG-NAM reproducibility and source-API preflight.

Purpose: capture the exact local revision code / environment needed to write
faithful, leakage-controlled native SIAN/NIT/iRF adapters. No training, no
external downloads, no modifications to existing benchmark outputs.

Run from the AG-NAM repository root:
    python scripts/collect_r12_preflight.py
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

MAX_FILE_BYTES = 350_000
MAX_ZIP_BYTES = 9_000_000
ALLOWED_EXTENSIONS = {'.py', '.yml', '.yaml', '.toml', '.md', '.json', '.txt', '.cfg', '.ini'}
SKIP_PARTS = {
    '.git', '.venv', 'venv', 'env', '__pycache__', 'site-packages',
    'results', 'data', 'datasets', 'outputs', 'checkpoints',
    '.pytest_cache', '.mypy_cache', 'wandb', 'mlruns', '.idea',
    '.vscode', 'node_modules', 'private', 'secrets', 'credentials',
}
SENSITIVE_NAME = re.compile(r'(token|password|credential|secret|api[_-]?key|\.env|private[_-]?key)', re.I)
MODULES = ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'sian', 'nampy',
           'psutil', 'pynvml', 'openml', 'catboost', 'interpret', 'irf', 'rpy2')
INTERESTING = re.compile(
    r'(synthetic|realx|external|revision|baseline|dataset|generator|benchmark|'
    r'pair|preprocess|interaction|pipeline|cost|runtime|split|protocol|manifest|'
    r'agnam|sian|train|fit|model|runner|metrics)', re.I
)


def run_safe(cmd: list[str], cwd: Path, timeout: int = 8) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout, encoding='utf-8', errors='replace')
        return {'returncode': p.returncode, 'stdout': p.stdout[:10000],
                'stderr': p.stderr[:1500]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'error': str(e)}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def package_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MODULES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        out[name] = version
    return out


def python_symbols(path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding='utf-8-sig', errors='replace'), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as e:
        return {'parse_error': str(e)}
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    imports: set[str] = set()
    cli_option_strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == 'add_argument':
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--'):
                        cli_option_strings.add(a.value)
    # Top-level and class methods only; not all nested functions.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if INTERESTING.search(node.name):
                functions.append({'name': node.name, 'line': node.lineno,
                                  'args': [x.arg for x in node.args.args]})
        elif isinstance(node, ast.ClassDef):
            if INTERESTING.search(node.name):
                classes.append({'name': node.name, 'line': node.lineno,
                                'methods': [x.name for x in node.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]})
    return {'classes': classes[:90], 'functions': functions[:150],
            'cli_options': sorted(cli_option_strings), 'imports': sorted(imports)[:180]}


def source_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for base in ('src/agnam', 'scripts', 'configs', 'tests', 'docs'):
        directory = root / base
        if not directory.is_dir():
            continue
        for p in directory.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            rel = p.relative_to(root)
            if any(part.lower() in SKIP_PARTS for part in rel.parts):
                continue
            if SENSITIVE_NAME.search(rel.name):
                continue
            if base == 'docs' and not INTERESTING.search(p.name):
                continue
            if base == 'tests' and not INTERESTING.search(p.name):
                continue
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            selected.append(p)
    for name in ('pyproject.toml', 'environment.yml', 'README.md', 'requirements.txt',
                 'requirements-dev.txt', 'CITATION.cff'):
        p = root / name
        if p.is_file() and p.stat().st_size <= MAX_FILE_BYTES:
            selected.append(p)
    return sorted(set(selected))


def existing_revision_evidence(root: Path) -> dict[str, Any]:
    targets = (
        'results/revision/external_interaction_baselines',
        'results/revision/external_isr_hybrid',
        'results/revision/attention_ablation',
        'results/revision/openml_calibration_r8',
        'results/revision/domain_grounded_r10',
        'results/realx/benchmark',
        'results/synthetic',
    )
    out: dict[str, Any] = {}
    for t in targets:
        p = root / t
        if not p.is_dir():
            out[t] = {'exists': False}
            continue
        counts: Counter[str] = Counter()
        for path in p.rglob('*'):
            if path.is_file():
                counts[path.suffix.lower() or '<none>'] += 1
        out[t] = {'exists': True, 'file_count': sum(counts.values()), 'extensions': dict(counts)}
    return out


def local_package_symbols(name: str) -> dict[str, Any]:
    """Only fingerprints/signatures of installed third-party packages; do not copy source."""
    answer: dict[str, Any] = {'installed': False}
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError) as e:
        return {'installed': False, 'error': str(e)}
    if spec is None:
        return answer
    answer['installed'] = True
    answer['origin'] = str(spec.origin) if spec.origin else None
    paths = list(spec.submodule_search_locations or [])
    files: list[Path] = []
    for base in paths:
        bp = Path(base)
        if bp.is_dir():
            for p in bp.rglob('*.py'):
                if ('test' not in {x.lower() for x in p.parts}) and p.stat().st_size <= MAX_FILE_BYTES:
                    files.append(p)
    rows = []
    for p in sorted(files)[:250]:
        summary = python_symbols(p)
        # Collect only relevant public names. No third-party source content.
        if summary.get('classes') or summary.get('functions'):
            rows.append({'module_file': str(p.relative_to(paths[0])) if paths else p.name,
                         'sha256': hash_file(p),
                         'classes': summary.get('classes', []),
                         'functions': summary.get('functions', []),
                         'cli_options': summary.get('cli_options', [])})
    answer['symbol_summaries'] = rows[:120]
    answer['source_file_count_scanned'] = len(files)
    return answer


def main() -> int:
    root = Path.cwd().resolve()
    if not (root / 'src' / 'agnam').is_dir():
        print('ERROR: execute in the AG-NAM repository root; src/agnam was not found.', file=sys.stderr)
        return 2
    output = root / 'AG_NAM_R12_Preflight.zip'
    files = source_files(root)
    env = {'generated_utc': datetime.now(timezone.utc).isoformat(),
           'purpose': 'read-only preflight, not an experiment or a validated baseline',
           'python': sys.version, 'python_executable': sys.executable,
           'platform': platform.platform(), 'machine': platform.machine(),
           'cpu_count': os.cpu_count(), 'installed_packages': package_versions(),
           'git_branch': run_safe(['git', 'branch', '--show-current'], root),
           'git_head': run_safe(['git', 'rev-parse', 'HEAD'], root),
           'git_status_short': run_safe(['git', 'status', '--short'], root),
           'gpu': run_safe(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                            '--format=csv,noheader'], root),
           'evidence_inventory': existing_revision_evidence(root),
           'third_party_symbol_catalog': {'sian': local_package_symbols('sian')},
           }
    symbols: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        manifest.append({'path': rel, 'sha256': hash_file(p), 'bytes': p.stat().st_size})
        if p.suffix.lower() == '.py':
            symbols[rel] = python_symbols(p)
    # ZIP only analyzed public-style repo files; never include datasets or credentials.
    report = {'environment': env, 'source_manifest': manifest,
              'symbol_catalog': symbols, 'training_performed': False,
              'benchmark_outputs_modified': False,
              'baseline_experiments_completed': []}
    added_bytes = 0
    with ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('R12_PREFLIGHT_REPORT.json', json.dumps(report, indent=2, ensure_ascii=False))
        for p in files:
            size = p.stat().st_size
            if added_bytes + size > MAX_ZIP_BYTES:
                continue
            rel = p.relative_to(root).as_posix()
            z.write(p, 'source_snapshot/' + rel)
            added_bytes += size
    with ZipFile(output) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('ZIP integrity failed: ' + bad)
    print('=== AG-NAM R12 READ-ONLY PREFLIGHT ===')
    print('REPOSITORY:', root)
    print('GIT BRANCH:', env['git_branch'].get('stdout', '').strip())
    print('PYTHON:', sys.version.split()[0])
    print('SIAN:', env['installed_packages']['sian'])
    print('SOURCE FILES:', len(files))
    print('ZIP MiB:', round(output.stat().st_size / 1048576, 3))
    print('ARCHIVE:', output)
    print('TRAINING: NO | BENCHMARK OUTPUTS MODIFIED: NO')
    print('NEXT: Upload AG_NAM_R12_Preflight.zip to this conversation.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
