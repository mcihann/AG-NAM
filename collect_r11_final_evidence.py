"""Collect AG-NAM R2–R11 editorial evidence into one small, auditable ZIP.

Usage (from AG-NAM root): python collect_r11_final_evidence.py
Read-only with respect to experiment/code trees. Does not run training.
The ZIP includes whitelisted publication/analysis summaries, not raw datasets or models.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path.cwd()
OUTPUT = ROOT / "AG_NAM_R11_Final_Evidence.zip"
MAX_FILE = 5 * 1024 * 1024  # CSV and prose summaries; excludes large raw artifacts.
MAX_TOTAL = 90 * 1024 * 1024
EXTS = {".csv", ".json", ".md", ".txt", ".tex", ".bib", ".docx"}


def source_files(root: Path) -> list[Path]:
    picks: set[Path] = set()

    def add(p: Path) -> None:
        if p.is_file() and p.suffix.lower() in EXTS:
            picks.add(p)

    for pattern in (
        "docs/revision_*.md", "docs/*protocol*.md", "docs/revision_*.txt",
        "results/synthetic/publication/*.csv", "results/synthetic/publication/*.txt",
        "results/synthetic/statistics/*.csv",
        "results/openml/publication/*.csv", "results/realx/benchmark/publication/**/*.csv",
        "results/revision/r11_publication_tables/*",
        "Manuscript.docx", "manuscript.docx", "Manuscript.tex", "manuscript.tex",
        "paper/main.tex", "paper/manuscript.tex", "paper/references.bib", "manuscript/references.bib",
    ):
        for p in root.glob(pattern):
            add(p)

    revision = root / "results" / "revision"
    if revision.is_dir():
        for module in revision.iterdir():
            if not module.is_dir():
                continue
            # Actual aggregate analyses, not hundreds of model checkpoints or raw datasets.
            for sub in ("analysis", "publication", "statistics", "final_analysis_v1", "tables"):
                area = module / sub
                if area.is_dir():
                    for p in area.rglob("*"):
                        add(p)
            for p in module.iterdir():
                if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
                    add(p)

        # Run-level summaries required to reproduce paired comparisons, not raw pair surfaces.
        for pattern in (
            "external_interaction_baselines/core_synthetic/S*/realization_*_summary.csv",
            "external_interaction_baselines/core_synthetic/S*/realization_*_manifest.json",
            "external_isr_hybrid/core_synthetic/S*/realization_*/*_summary.csv",
            "external_interaction_baselines/realx/**/*.csv",
            "external_interaction_baselines/realx/**/*.json",
            "external_interaction_baselines/real_x/**/*.csv",
            "external_interaction_baselines/real_x/**/*.json",
            "external_interaction_baselines/additive_groves/S1/realization_0[1-5]/additive_groves_summary.csv",
            "external_interaction_baselines/additive_groves/S1/realization_0[1-5]/additive_groves_manifest.json",
        ):
            for p in revision.glob(pattern):
                add(p)
    return sorted(picks, key=lambda x: x.relative_to(root).as_posix())


def build(root: Path, output: Path) -> tuple[int, dict]:
    if not (root / "results" / "revision").is_dir():
        raise SystemExit("ERROR: Run from the AG-NAM repository root (results/revision missing).")

    chosen: list[Path] = []
    skipped = []
    total = 0
    for p in source_files(root):
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        if size > MAX_FILE or total + size > MAX_TOTAL:
            skipped.append({"path": rel, "bytes": size, "reason": "size_guard"})
        else:
            chosen.append(p)
            total += size

    entries = []
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as z:
        for p in chosen:
            rel = p.relative_to(root).as_posix()
            raw = p.read_bytes()
            z.writestr(rel, raw)
            entries.append({"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        sections = Counter(e["path"].split("/")[2] if e["path"].startswith("results/revision/") and len(e["path"].split("/")) > 2 else e["path"].split("/")[0] for e in entries)
        manifest = {
            "purpose": "AG-NAM final R11 manuscript evidence audit",
            "read_only_source_collection": True,
            "no_training_or_experiments_executed": True,
            "included_count": len(entries),
            "included_original_bytes": total,
            "categories": dict(sorted(sections.items())),
            "skipped": skipped,
            "files": entries,
            "caution": "Archive does not prove experimental claims; original per-trial outputs and protocols must be cross-checked before manuscript submission.",
        }
        z.writestr("R11_COLLECTION_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    with ZipFile(output) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
    return len(entries), manifest


if __name__ == "__main__":
    n, manifest = build(ROOT, OUTPUT)
    print("=== AG-NAM R11 FINAL EVIDENCE COLLECTION ===")
    print("ARCHIVE:", OUTPUT.resolve())
    print("FILES:", n)
    print("ZIP MiB:", round(OUTPUT.stat().st_size / (1024 * 1024), 2))
    print("INCLUDED MODULES:", json.dumps(manifest["categories"], ensure_ascii=False, sort_keys=True))
    print("SIZE-GUARD SKIPS:", len(manifest["skipped"]))
    print("Training: NO | Existing results modified: NO")
    print("NEXT: Upload AG_NAM_R11_Final_Evidence.zip to this conversation")
