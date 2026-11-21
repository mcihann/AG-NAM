from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import traceback


MODULES = (
    "sian.interpret.explainer2",
    "sian.interpret.basic_wrappers",
    "sian.models.models",
    "sian.models.training",
    "sian.experiment.experiment_mixins",
    "sian.experiment.experiment",
)

KEYWORDS = (
    "arch",
    "explain",
    "wrap",
    "model",
    "mlp",
    "train",
    "fit",
    "fis",
    "interaction",
)


def public_members(module):
    rows = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        module_name = getattr(obj, "__module__", "")
        if module_name != module.__name__:
            continue
        try:
            signature = str(inspect.signature(obj))
        except Exception:
            signature = "<signature unavailable>"
        rows.append((name, type(obj).__name__, signature))
    return rows


def should_dump_source(name: str) -> bool:
    low = name.lower()
    return any(keyword in low for keyword in KEYWORDS)


def main():
    output_dir = Path(
        "results/revision/external_interaction_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "sian_runtime_api_audit.txt"
    json_path = output_dir / "sian_runtime_api_audit.json"

    report = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "sian_runtime_api_audit",
        "modules": {},
    }

    chunks = []

    for module_name in MODULES:
        chunks.append("=" * 100)
        chunks.append(f"MODULE: {module_name}")
        chunks.append("=" * 100)

        module_report = {
            "import_ok": False,
            "file": None,
            "error": None,
            "members": [],
        }

        try:
            module = importlib.import_module(module_name)
            module_report["import_ok"] = True
            module_report["file"] = getattr(module, "__file__", None)

            chunks.append(f"FILE: {module_report['file']}")
            chunks.append("")

            members = public_members(module)
            module_report["members"] = [
                {
                    "name": name,
                    "kind": kind,
                    "signature": signature,
                }
                for name, kind, signature in members
            ]

            chunks.append("PUBLIC CALLABLES:")
            for name, kind, signature in members:
                chunks.append(
                    f"  {kind:10s} {name}{signature}"
                )

            chunks.append("")
            chunks.append("SELECTED SOURCE:")
            chunks.append("")

            for name, kind, signature in members:
                if not should_dump_source(name):
                    continue
                obj = getattr(module, name)
                chunks.append("-" * 80)
                chunks.append(f"{kind.upper()}: {name}{signature}")
                chunks.append("-" * 80)
                try:
                    chunks.append(inspect.getsource(obj))
                except Exception as exc:
                    chunks.append(
                        f"<source unavailable: "
                        f"{type(exc).__name__}: {exc}>"
                    )
                chunks.append("")

        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
            )
            module_report["error"] = error
            chunks.append(f"IMPORT FAILED: {error}")
            chunks.append("")
            chunks.append(traceback.format_exc())

        report["modules"][module_name] = module_report
        chunks.append("")

    txt_path.write_text(
        "\n".join(chunks),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=== SIAN RUNTIME API AUDIT ===")
    for module_name, module_report in report["modules"].items():
        print(
            f"{module_name}: "
            f"import_ok={module_report['import_ok']}"
        )
        if module_report["error"]:
            print(f"  {module_report['error']}")

    print()
    print("Saved:")
    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    main()
