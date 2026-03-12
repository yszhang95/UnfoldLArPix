#!/usr/bin/env python3
"""
Deconvolution analysis pipeline for LArPix data.

Steps
-----
  1  Run deconv_positron_v1.py and/or v2.py for all (sigma, sigma_pxl) combinations.
  2  Export wire-cell JSON files at each threshold via deconv_xyz.py,
     including smeared-true JSONs.
  3  Copy all JSONs to --dest-dir.
  4  Generate histogram plots via plot_proj.py for each configuration.

Usage examples
--------------
  # Full pipeline with defaults
  python run_analysis.py

  # Custom sigma grid, only V2, steps 1-3
  python run_analysis.py --version v2 --sigmas 0.005 0.01 --sigma-pxls 0.1 0.2 \\
      --steps 1 2 3

  # Only regenerate plots (step 4) with existing npz files
  python run_analysis.py --steps 4

  # Dry run to see what commands would execute
  python run_analysis.py --dry-run
"""

import argparse
import subprocess
import sys
import shutil
from itertools import product
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_sigma(v: float) -> str:
    """0.005 -> '005', 0.01 -> '010', 0.2 -> '200'"""
    return f"{v:.3f}".lstrip("0").replace(".", "")


def fmt_sigma_pxl(v: float) -> str:
    """0.1 -> '10', 0.15 -> '15', 0.2 -> '20'"""
    return f"{v:.2f}".lstrip("0").replace(".", "")


def fmt_threshold(v: float) -> str:
    """1.5 -> '1p5', 0.5 -> '0p5'"""
    return str(v).replace(".", "p")


def run(cmd: list[str], dry: bool, cwd: Path) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    if not dry:
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            sys.exit(f"Command failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_deconv(cfg, cwd: Path, dry: bool) -> None:
    """Run deconv scripts for every (sigma, sigma_pxl) pair."""
    print("\n=== Step 1: Deconvolution ===")
    for sigma, sigma_pxl in product(cfg.sigmas, cfg.sigma_pxls):
        print(f"\n  sigma={sigma}  sigma_pxl={sigma_pxl}")
        for ver in cfg.versions:
            script = f"deconv_positron_{ver}.py"
            run([sys.executable, script,
                 "--sigma", str(sigma),
                 "--sigma-pxl", str(sigma_pxl),
                 "--input-file", cfg.input_file,
                 "--field-response", cfg.field_response],
                dry, cwd)


def step2_export(cfg, cwd: Path, dry: bool) -> None:
    """Export JSON files (deconv + smeared) for every combination x threshold."""
    print("\n=== Step 2: JSON export ===")
    out_dir = cfg.output_matrix
    for sigma, sigma_pxl in product(cfg.sigmas, cfg.sigma_pxls):
        ss = fmt_sigma(sigma)
        sp = fmt_sigma_pxl(sigma_pxl)
        for ver in cfg.versions:
            npz = f"deconv_positron{'_v2' if ver == 'v2' else ''}_event_0_0.npz"
            for thr in cfg.thresholds:
                ts = fmt_threshold(thr)
                prefix = f"{ver}_s{ss}_sp{sp}_t{ts}"
                run([sys.executable, "deconv_xyz.py", npz,
                     "--tpc-id", "0", "--event-id", "0",
                     "--threshold", str(thr),
                     "--prefix", prefix,
                     "--smeared-prefix", f"{prefix}_smeared",
                     "--output-dir", str(out_dir)],
                    dry, cwd)


def step3_copy(cfg, cwd: Path, dry: bool) -> None:
    """Copy all exported JSONs to dest-dir."""
    print(f"\n=== Step 3: Copy to {cfg.dest_dir} ===")
    dest = Path(cfg.dest_dir)
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
    for sigma, sigma_pxl in product(cfg.sigmas, cfg.sigma_pxls):
        ss = fmt_sigma(sigma)
        sp = fmt_sigma_pxl(sigma_pxl)
        for ver in cfg.versions:
            for thr in cfg.thresholds:
                ts = fmt_threshold(thr)
                prefix = f"{ver}_s{ss}_sp{sp}_t{ts}"
                src_dir = Path(cfg.output_matrix) / "data" / "0"
                for suffix in ("", "_smeared"):
                    src = src_dir / f"0-{prefix}{suffix}.json"
                    dst = dest / f"0-{prefix}{suffix}.json"
                    print(f"  cp {src.name} -> {dest}/")
                    if not dry:
                        shutil.copy2(src, dst)


def step4_plots(cfg, cwd: Path, dry: bool) -> None:
    """Generate histogram plots for every (sigma, sigma_pxl) x version."""
    print(f"\n=== Step 4: Plots -> {cfg.plot_dir} ===")
    plot_dir = Path(cfg.plot_dir)
    if not dry:
        plot_dir.mkdir(parents=True, exist_ok=True)
    for sigma, sigma_pxl in product(cfg.sigmas, cfg.sigma_pxls):
        ss = fmt_sigma(sigma)
        sp = fmt_sigma_pxl(sigma_pxl)
        for ver in cfg.versions:
            npz = f"deconv_positron{'_v2' if ver == 'v2' else ''}_event_0_0.npz"
            prefix = plot_dir / f"{ver}_s{ss}_sp{sp}"
            run([sys.executable, "plot_proj.py", npz,
                 "--threshold", str(cfg.plot_threshold),
                 "--prefix", str(prefix)],
                dry, cwd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LArPix deconvolution analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.005, 0.01],
                   metavar="S", help="Temporal sigma values for deconv regularisation")
    p.add_argument("--sigma-pxls", nargs="+", type=float, default=[0.1, 0.15, 0.2],
                   metavar="SP", help="Pixel sigma values for deconv regularisation")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                   metavar="T", help="Amplitude thresholds for JSON export (ke-)")
    p.add_argument("--versions", nargs="+", choices=["v1", "v2"], default=["v1", "v2"],
                   help="Processor versions to run")
    p.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4], metavar="N",
                   help="Which pipeline steps to run (1=deconv 2=export 3=copy 4=plots)")
    p.add_argument("--output-matrix", default="output_matrix",
                   help="Intermediate JSON output directory for deconv_xyz")
    p.add_argument("--dest-dir", default="raw_positron/data/0",
                   help="Final destination directory for JSON files (step 3)")
    p.add_argument("--plot-dir", default="plots",
                   help="Output directory for histogram plots (step 4)")
    p.add_argument("--plot-threshold", type=float, default=0.5,
                   help="Threshold passed to plot_proj.py")
    p.add_argument("--input-file",
                   default="data/pgun_positron_3gev_tred_noises_effq_nt1.npz",
                   help="Input NPZ file produced by tred (passed to deconv scripts)")
    p.add_argument("--field-response",
                   default="/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz",
                   help="Field response NPZ file (passed to deconv scripts)")
    p.add_argument("--cwd", default=".",
                   help="Working directory (directory containing the scripts)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them")
    return p.parse_args()


def main():
    cfg = parse_args()
    cwd = Path(cfg.cwd).resolve()

    step_map = {
        1: step1_deconv,
        2: step2_export,
        3: step3_copy,
        4: step4_plots,
    }

    print(f"Pipeline: steps={cfg.steps}  versions={cfg.versions}")
    print(f"  sigmas={cfg.sigmas}  sigma_pxls={cfg.sigma_pxls}")
    print(f"  thresholds={cfg.thresholds}")
    print(f"  cwd={cwd}  dry={cfg.dry_run}")

    for step_num in sorted(cfg.steps):
        step_map[step_num](cfg, cwd, cfg.dry_run)

    print("\n=== Done ===")
    if not cfg.dry_run:
        dest = Path(cfg.dest_dir)
        if dest.exists():
            n = len(list(dest.glob("*.json")))
            print(f"  JSON files in {dest}: {n}")
        plot_dir = Path(cfg.plot_dir)
        if plot_dir.exists():
            n = len(list(plot_dir.glob("*.png")))
            print(f"  Plots in {plot_dir}: {n}")


if __name__ == "__main__":
    main()
