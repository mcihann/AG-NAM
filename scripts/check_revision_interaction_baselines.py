from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def package_info(import_name: str, distribution_name: str | None = None):
    distribution_name = distribution_name or import_name
    result = {
        "import_name": import_name,
        "distribution_name": distribution_name,
        "installed": False,
        "import_ok": False,
        "version": None,
        "error": None,
    }

    try:
        result["version"] = importlib.metadata.version(distribution_name)
        result["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as exc:
        result["error"] = f"metadata: {type(exc).__name__}: {exc}"

    try:
        importlib.import_module(import_name)
        result["import_ok"] = True
    except Exception as exc:
        result["error"] = f"import: {type(exc).__name__}: {exc}"

    return result


def command_version(command: str, args: list[str]):
    path = shutil.which(command)
    result = {
        "command": command,
        "path": path,
        "available": path is not None,
        "version_output": None,
        "error": None,
    }
    if path is None:
        return result

    try:
        proc = subprocess.run(
            [path, *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        result["version_output"] = output[:2000]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def find_treeextra():
    candidates = []

    env_path = os.environ.get("TREEEXTRA_HOME")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path.cwd() / "external" / "TreeExtra",
            Path.cwd() / "tools" / "TreeExtra",
            Path.home() / "TreeExtra",
        ]
    )

    executable_names = (
        "ag_interactions.exe",
        "ag_interactions",
    )

    hits = []
    for base in candidates:
        for name in executable_names:
            path = base / name
            if path.exists():
                hits.append(str(path.resolve()))

    path_hit = shutil.which("ag_interactions")
    if path_hit:
        hits.append(str(Path(path_hit).resolve()))

    path_hit_exe = shutil.which("ag_interactions.exe")
    if path_hit_exe:
        hits.append(str(Path(path_hit_exe).resolve()))

    return {
        "treeextra_home_env": env_path,
        "ag_interactions_hits": sorted(set(hits)),
        "available": bool(hits),
    }


def main():
    report = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "external_interaction_baseline_environment_audit",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "core_packages": {
            "numpy": package_info("numpy"),
            "pandas": package_info("pandas"),
            "sklearn": package_info("sklearn", "scikit-learn"),
            "torch": package_info("torch"),
            "scipy": package_info("scipy"),
            "psutil": package_info("psutil"),
        },
        "baseline_packages": {
            "sian": package_info("sian"),
            "gaminet_optional": package_info("gaminet"),
            "irf_python": package_info("irf"),
        },
        "external_runtimes": {
            "Rscript": command_version("Rscript", ["--version"]),
            "git": command_version("git", ["--version"]),
        },
        "treeextra": find_treeextra(),
        "policy": {
            "do_not_downgrade_main_environment_for_irf": True,
            "nit_not_substituted_by_nid": True,
            "requested_baselines": [
                "friedman_h",
                "nit",
                "additive_groves",
                "irf",
                "sian",
            ],
            "secondary_baselines": ["nid"],
        },
    }

    output = Path(
        "results/revision/external_interaction_baselines/"
        "environment_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== EXTERNAL INTERACTION BASELINE ENVIRONMENT AUDIT ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print()

    for name, info in report["core_packages"].items():
        print(
            f"{name:12s} installed={str(info['installed']):5s} "
            f"import_ok={str(info['import_ok']):5s} "
            f"version={info['version']}"
        )

    print()
    for name, info in report["baseline_packages"].items():
        print(
            f"{name:16s} installed={str(info['installed']):5s} "
            f"import_ok={str(info['import_ok']):5s} "
            f"version={info['version']}"
        )
        if info["error"]:
            print(f"  error: {info['error']}")

    print()
    r = report["external_runtimes"]["Rscript"]
    print(f"Rscript available={r['available']} path={r['path']}")

    tree = report["treeextra"]
    print(
        "TreeExtra ag_interactions available="
        f"{tree['available']}"
    )
    for hit in tree["ag_interactions_hits"]:
        print(f"  {hit}")

    print()
    print("Saved:")
    print(output)


if __name__ == "__main__":
    main()
