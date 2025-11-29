"""Export all .ipynb notebooks in exercise folders to individual PDFs and merge.

Usage (PowerShell):
    python scripts/export_notebooks.py \
        --source . \
        --pattern "ex*/**/*.ipynb" \
        --export-dir exports \
        --combined-pdf combined_lab_reports.pdf

Requires: nbconvert[webpdf], PyPDF2
Optional: For LaTeX PDF (if webpdf fails) install MiKTeX or TeXLive.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from traitlets.config import Config

# Lazy imports inside functions for clearer error messages

def find_notebooks(source: Path, pattern: str) -> List[Path]:
    notebooks = [p for p in source.glob(pattern) if p.is_file() and not p.name.startswith(".~")]
    # Exclude checkpoint directories
    notebooks = [p for p in notebooks if ".ipynb_checkpoints" not in p.parts]
    return sorted(notebooks)


def export_webpdf(nb_path: Path, out_dir: Path) -> Path:
    try:
        from nbconvert import WebPDFExporter  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "nbconvert WebPDFExporter not available. Install with 'pip install nbconvert[webpdf]'"
        ) from e

    c = Config()
    c.WebPDFExporter.allow_chromium_download = True  # auto-download Chromium if needed
    exporter = WebPDFExporter(config=c)
    exporter.exclude_input = False
    pdf_data, _ = exporter.from_filename(str(nb_path))
    out_path = out_dir / f"{nb_path.stem}.pdf"
    out_path.write_bytes(pdf_data)
    return out_path


def merge_pdfs(pdf_paths: List[Path], merged_path: Path) -> None:
    from PyPDF2 import PdfMerger  # type: ignore

    merger = PdfMerger()
    for pdf in pdf_paths:
        merger.append(str(pdf))
    merger.write(str(merged_path))
    merger.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export notebooks to PDF and merge into one.")
    parser.add_argument("--source", default=".", help="Root directory to scan")
    parser.add_argument(
        "--pattern", default="ex*/**/*.ipynb", help="Glob pattern relative to source root"
    )
    parser.add_argument("--export-dir", default="exports", help="Directory for individual PDFs")
    parser.add_argument(
        "--combined-pdf", default="combined_lab_reports.pdf", help="Filename for merged PDF"
    )
    args = parser.parse_args(argv)

    source_root = Path(args.source).resolve()
    export_dir = Path(args.export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    notebooks = find_notebooks(source_root, args.pattern)
    if not notebooks:
        print("No notebooks found.")
        return 1

    print(f"Found {len(notebooks)} notebooks. Exporting to '{export_dir}'.")

    exported = []
    for nb in notebooks:
        print(f"[EXPORT] {nb}")
        try:
            pdf_path = export_webpdf(nb, export_dir)
            exported.append(pdf_path)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Failed webpdf export for {nb}: {e}")
            print("Attempting LaTeX PDF fallback (requires TeX distribution)...")
            try:
                from nbconvert import PDFExporter  # type: ignore

                exporter = PDFExporter()
                exporter.exclude_input = False
                pdf_data, _ = exporter.from_filename(str(nb))
                pdf_path = export_dir / f"{nb.stem}.pdf"
                pdf_path.write_bytes(pdf_data)
                exported.append(pdf_path)
            except Exception as e2:  # noqa: BLE001
                print(f"[FATAL] Could not export {nb}: {e2}")

    if not exported:
        print("No PDFs exported; aborting merge.")
        return 2

    merged_path = export_dir / args.combined_pdf
    print(f"Merging {len(exported)} PDFs into {merged_path}")
    try:
        merge_pdfs(exported, merged_path)
    except Exception as e:  # noqa: BLE001
        print(f"Merge failed: {e}")
        return 3

    print("Export + merge complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
