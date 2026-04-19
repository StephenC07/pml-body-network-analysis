#!/usr/bin/env python3
"""
PML body internal network analysis from a cryo-electron tomogram.

Given a reconstructed tomogram (MRC) and a paired PML body
segmentation (multi-label TIFF), the pipeline:

    1. Normalises each Z-slice independently (z-score, clip to
       [-3, +3] standard deviations, linear rescale to [0, 1]).
    2. Denoises with total-variation Chambolle filtering.
    3. Builds a bright-region exclusion mask from Gaussian-smoothed
       intensity (excludes the top `--bright-pct` percent of pixels).
    4. Enhances local contrast with CLAHE.
    5. Extracts dark fine features with a black top-hat transform.
    6. Thresholds at the `--thresh-pct` percentile of the unmasked
       pixels and cleans the binary mask.
    7. Skeletonises (Zhang-Suen) and identifies branch points
       (skeleton pixels with at least three neighbours).
    8. Quantifies the network on a `--grid-n`-by-`--grid-n` tile grid,
       combining fibre density and branch-point density into a single
       composite connectivity score (60 percent fibre + 40 percent
       branches).
    9. Computes the Pearson correlation between per-cell connectivity
       and per-cell PML body volume fraction.
   10. In parallel, computes the Euclidean distance of "dense"
       pixels (bottom `--dense-pct` percent after a large Gaussian
       blur) from the PML body surface, and compares that distribution
       to a random background sample.

Outputs (written to --output-dir):

    summary.json
        Parameters used, per-grid-cell means, Pearson correlation,
        median distances.
    connectivity_grid.csv
        grid_n x grid_n composite connectivity score.
    fiber_density_grid.csv
        grid_n x grid_n normalised fibre density.
    branch_density_grid.csv
        grid_n x grid_n normalised branch-point density.
    pml_volume_fraction_grid.csv
        grid_n x grid_n PML volume fraction.
    distances_dense_px.csv, distances_background_px.csv
        Euclidean distance (bin-4 pixels) from the PML body surface.
    diagnostic_figure.png
        Four-panel visual summary for quality control.

Reference:
    Pražák V et al. In situ molecular architecture of eYFP-PML.I
    bodies by cryo-CLEM and cryo-electron tomography reveals
    trinucleosomes in an open state.

Example:
    python pml_network_analysis.py \\
        --tomogram tomogram.mrc \\
        --segmentation pml_labels.tif \\
        --pml-label 2 \\
        --pixel-size-nm 1.0 \\
        --output-dir results/ \\
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import tifffile
from scipy.ndimage import convolve, distance_transform_edt
from skimage import exposure, morphology, restoration
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, black_tophat, disk, skeletonize


# Parameters exposed via CLI; also used as defaults in the functions
# below so that the module can be imported and reused programmatically.
DEFAULTS = {
    "grid_n": 12,
    "bright_pct": 75,
    "thresh_pct": 80,
    "dense_pct": 15,
    "dense_sigma": 20.0,
    "dense_min_size": 500,
    "tv_weight": 0.05,
    "bright_sigma": 15.0,
    "clahe_clip": 0.02,
    "clahe_kernel": 64,
    "tophat_radius": 6,
    "connect_radius": 2,
    "min_object": 10,
    "fill_hole": 50,
    "fiber_weight": 0.6,
    "branch_weight": 0.4,
    "random_seed": 42,
    "dense_z_step": 3,
    "dense_subsample": 10,
    "bg_sample_per_slice": 500,
}


# ────────────────────────────────────────────────────────────────────
# Per-slice skeleton pipeline
# ────────────────────────────────────────────────────────────────────

def normalise_slice(sl: np.ndarray) -> np.ndarray:
    """z-score, clip to +/-3 sigma, rescale to [0, 1]."""
    m, s = sl.mean(), sl.std()
    z = np.clip((sl - m) / (s + 1e-12), -3, 3)
    return (z + 3) / 6.0


def segment_slice(
    sl: np.ndarray,
    bright_pct: float = DEFAULTS["bright_pct"],
    thresh_pct: float = DEFAULTS["thresh_pct"],
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full 2D segmentation pipeline on one slice.

    Returns (skeleton, branch_points), both boolean arrays with the
    shape of the input slice.
    """
    sl_norm = normalise_slice(sl.astype(np.float32))
    sl_denoised = restoration.denoise_tv_chambolle(
        sl_norm, weight=DEFAULTS["tv_weight"]
    )
    bright_bg = gaussian(sl_denoised, sigma=DEFAULTS["bright_sigma"])
    bright_mask = bright_bg > np.percentile(bright_bg, bright_pct)

    sl_clahe = exposure.equalize_adapthist(
        sl_denoised,
        clip_limit=DEFAULTS["clahe_clip"],
        kernel_size=DEFAULTS["clahe_kernel"],
    )

    bth = black_tophat(sl_clahe, disk(DEFAULTS["tophat_radius"]))
    bth_n = (bth - bth.min()) / (bth.max() - bth.min() + 1e-12)
    thresh = np.percentile(bth_n[~bright_mask], thresh_pct)
    bth_n[bright_mask] = 0

    binary = bth_n > thresh
    binary = morphology.remove_small_objects(binary, min_size=DEFAULTS["min_object"])
    connected = binary_dilation(binary, disk(DEFAULTS["connect_radius"]))
    connected = morphology.remove_small_holes(connected, area_threshold=DEFAULTS["fill_hole"])

    skel = skeletonize(connected)
    # 3x3 convolution counts the pixel plus its 8 neighbours; a skeleton
    # pixel with k true neighbours has convolution value k+1.  Branch
    # points are defined here as pixels with at least 3 neighbours
    # (convolution value >= 4) as used by Haralick & Shapiro.
    neighbour_sum = convolve(skel.astype(np.int8), np.ones((3, 3), dtype=np.int8))
    branches = (neighbour_sum >= 4) & skel
    return skel, branches


def _segment_slice_worker(args):
    """Pickleable wrapper for multiprocessing.Pool.imap."""
    sl, bright_pct, thresh_pct = args
    return segment_slice(sl, bright_pct, thresh_pct)


def run_skeleton_volume(
    volume: np.ndarray,
    workers: int = 1,
    bright_pct: float = DEFAULTS["bright_pct"],
    thresh_pct: float = DEFAULTS["thresh_pct"],
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment every slice of a 3D volume.  Optionally parallel."""
    nz = volume.shape[0]
    skel_stack = np.zeros(volume.shape, dtype=bool)
    branch_stack = np.zeros(volume.shape, dtype=bool)
    slices = ((volume[i], bright_pct, thresh_pct) for i in range(nz))

    if workers > 1:
        with Pool(workers) as pool:
            for i, (skel, br) in enumerate(
                pool.imap(_segment_slice_worker, slices, chunksize=4)
            ):
                skel_stack[i] = skel
                branch_stack[i] = br
                if progress and i % 40 == 0:
                    print(f"    Z = {i}/{nz}", flush=True)
    else:
        for i, args in enumerate(slices):
            skel, br = _segment_slice_worker(args)
            skel_stack[i] = skel
            branch_stack[i] = br
            if progress and i % 40 == 0:
                print(f"    Z = {i}/{nz}", flush=True)

    return skel_stack, branch_stack


# ────────────────────────────────────────────────────────────────────
# Grid-based connectivity
# ────────────────────────────────────────────────────────────────────

def build_grid_connectivity(
    skel_stack: np.ndarray,
    branch_stack: np.ndarray,
    grid_n: int = DEFAULTS["grid_n"],
    fiber_weight: float = DEFAULTS["fiber_weight"],
    branch_weight: float = DEFAULTS["branch_weight"],
) -> dict:
    """Compute per-tile fibre and branch densities and the composite
    connectivity score used in the paper (60% fibre + 40% branch)."""
    nz, ny, nx = skel_stack.shape
    gh, gw = ny // grid_n, nx // grid_n
    cell_voxels = gh * gw * nz

    fiber = np.zeros((grid_n, grid_n))
    branch = np.zeros((grid_n, grid_n))
    for i in range(grid_n):
        for j in range(grid_n):
            r0, r1 = i * gh, (i + 1) * gh
            c0, c1 = j * gw, (j + 1) * gw
            fiber[i, j] = skel_stack[:, r0:r1, c0:c1].sum() / cell_voxels
            branch[i, j] = branch_stack[:, r0:r1, c0:c1].sum() / cell_voxels

    fiber_n = (fiber - fiber.min()) / (fiber.max() - fiber.min() + 1e-12)
    branch_n = (branch - branch.min()) / (branch.max() - branch.min() + 1e-12)
    composite = fiber_weight * fiber_n + branch_weight * branch_n
    return dict(
        fiber=fiber, branch=branch,
        fiber_norm=fiber_n, branch_norm=branch_n,
        composite=composite, gh=gh, gw=gw,
    )


def build_pml_volume_grid(
    pml_mask: np.ndarray, gh: int, gw: int, grid_n: int
) -> np.ndarray:
    """Per-tile PML body volume fraction.

    Computed only over Z-slices that actually contain PML voxels so
    that tomograms with partial PML coverage are not artificially
    diluted."""
    z_idx = np.where(pml_mask.sum(axis=(1, 2)) > 0)[0]
    if len(z_idx) == 0:
        return np.zeros((grid_n, grid_n))
    z0, z1 = z_idx[0], z_idx[-1]
    sub = pml_mask[z0:z1 + 1]
    nz = sub.shape[0]
    out = np.zeros((grid_n, grid_n))
    for i in range(grid_n):
        for j in range(grid_n):
            r0, r1 = i * gh, (i + 1) * gh
            c0, c1 = j * gw, (j + 1) * gw
            out[i, j] = sub[:, r0:r1, c0:c1].sum() / (gh * gw * nz)
    return out


# ────────────────────────────────────────────────────────────────────
# Dense region exclusion analysis
# ────────────────────────────────────────────────────────────────────

def dense_mask_slice(
    sl: np.ndarray,
    sigma: float = DEFAULTS["dense_sigma"],
    pct: float = DEFAULTS["dense_pct"],
    min_size: int = DEFAULTS["dense_min_size"],
    artifact_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Dense-region mask: bottom `pct`% of a Gaussian-smoothed slice."""
    sl_norm = normalise_slice(sl.astype(np.float32))
    bg = gaussian(sl_norm, sigma=sigma)
    bg_n = (bg - bg.min()) / (bg.max() - bg.min() + 1e-12)
    mask = bg_n < np.percentile(bg_n, pct)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_size)
    if artifact_mask is not None:
        mask[artifact_mask] = False
    return mask


def dense_distance_analysis(
    volume: np.ndarray,
    pml_mask: np.ndarray,
    z_step: int = DEFAULTS["dense_z_step"],
    dense_subsample: int = DEFAULTS["dense_subsample"],
    bg_sample_per_slice: int = DEFAULTS["bg_sample_per_slice"],
    seed: int = DEFAULTS["random_seed"],
    artifact_mask: np.ndarray | None = None,
) -> dict:
    """Compare distance-from-PML-surface for dense vs random bg pixels."""
    rng = np.random.default_rng(seed)
    pml_dist = distance_transform_edt(~pml_mask)
    z_idx = np.where(pml_mask.sum(axis=(1, 2)) > 0)[0]
    if len(z_idx) == 0:
        return dict(dense=np.array([]), background=np.array([]))
    z0, z1 = z_idx[0], z_idx[-1]

    dense, bg = [], []
    for zi in range(z0, z1 + 1, z_step):
        dm = dense_mask_slice(volume[zi], artifact_mask=artifact_mask)
        dense_pts = np.argwhere(dm)
        if len(dense_pts) > 0:
            for r, c in dense_pts[::dense_subsample]:
                dense.append(pml_dist[zi, r, c])

        exclude = ~dm & ~pml_mask[zi]
        if artifact_mask is not None:
            exclude &= ~artifact_mask
        bg_pts = np.argwhere(exclude)
        if len(bg_pts) > bg_sample_per_slice:
            idx = rng.choice(len(bg_pts), bg_sample_per_slice, replace=False)
            for r, c in bg_pts[idx]:
                bg.append(pml_dist[zi, r, c])

    return dict(dense=np.array(dense), background=np.array(bg))


# ────────────────────────────────────────────────────────────────────
# Diagnostic figure
# ────────────────────────────────────────────────────────────────────

def plot_diagnostic(
    volume: np.ndarray,
    pml_mask: np.ndarray,
    grid: dict,
    pml_grid: np.ndarray,
    distances: dict,
    pearson_r: float,
    pixel_size_nm: float,
    out_path: Path,
) -> None:
    """Four-panel QC figure:
        A) connectivity heatmap with PML outline
        B) PML volume fraction grid
        C) scatter: connectivity vs PML volume fraction
        D) distance-from-PML histogram (dense vs background)
    """
    ny, nx = volume.shape[1:]
    gh, gw = grid["gh"], grid["gw"]
    grid_n = grid["composite"].shape[0]

    mean_img = volume.mean(axis=0)
    m, s = mean_img.mean(), mean_img.std()
    mean_disp = (np.clip((mean_img - m) / (s + 1e-12), -1, 1) + 1) / 2
    pml_proj = pml_mask.max(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("PML body internal network: QC diagnostic", fontsize=13, fontweight="bold")

    # A. Connectivity heatmap overlay
    conn_full = np.kron(grid["composite"], np.ones((gh, gw)))[:ny, :nx]
    axes[0, 0].imshow(mean_disp, cmap="gray", alpha=0.4)
    im = axes[0, 0].imshow(conn_full, cmap="RdYlGn", alpha=0.6, vmin=0, vmax=1)
    axes[0, 0].contour(pml_proj, levels=[0.5], colors=["magenta"], linewidths=2)
    axes[0, 0].set_title(f"A. Connectivity score  ({grid_n}x{grid_n} grid)")
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.7, label="Connectivity")

    # B. PML volume fraction
    pml_full = np.kron(pml_grid, np.ones((gh, gw)))[:ny, :nx]
    axes[0, 1].imshow(mean_disp, cmap="gray", alpha=0.4)
    im2 = axes[0, 1].imshow(pml_full, cmap="Purples", alpha=0.7)
    axes[0, 1].contour(pml_proj, levels=[0.5], colors=["magenta"], linewidths=2)
    axes[0, 1].set_title("B. PML volume fraction per cell")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.7, label="PML fraction")

    # C. Scatter
    conn_flat = grid["composite"].ravel()
    pml_flat = pml_grid.ravel()
    has_pml = pml_flat > 0
    axes[1, 0].scatter(pml_flat[~has_pml] * 100, conn_flat[~has_pml],
                       c="gray", alpha=0.5, s=40, label="No PML")
    axes[1, 0].scatter(pml_flat[has_pml] * 100, conn_flat[has_pml],
                       c="magenta", alpha=0.7, s=60, label="Contains PML")
    if pml_flat.max() > 0:
        p = np.polyfit(pml_flat * 100, conn_flat, 1)
        xfit = np.linspace(0, pml_flat.max() * 100, 50)
        axes[1, 0].plot(xfit, np.polyval(p, xfit), "k--", alpha=0.6, lw=1.5)
    axes[1, 0].set_xlabel("PML volume fraction (%)")
    axes[1, 0].set_ylabel("Connectivity score")
    axes[1, 0].set_title(f"C. Correlation  r = {pearson_r:.3f}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # D. Distance histogram
    dense = distances["dense"]
    bg = distances["background"]
    if len(dense) and len(bg):
        max_px = max(dense.max(), bg.max())
        bins = np.arange(0, max_px + 10, 5)
        axes[1, 1].hist(bg * pixel_size_nm, bins=bins * pixel_size_nm,
                        alpha=0.4, color="gray", density=True, label="Background")
        axes[1, 1].hist(dense * pixel_size_nm, bins=bins * pixel_size_nm,
                        alpha=0.6, color="dodgerblue", density=True, label="Dense regions")
        axes[1, 1].axvline(np.median(dense) * pixel_size_nm, color="dodgerblue",
                           ls="--", lw=2,
                           label=f"Dense median = {np.median(dense) * pixel_size_nm:.0f} nm")
        axes[1, 1].axvline(np.median(bg) * pixel_size_nm, color="gray",
                           ls="--", lw=2,
                           label=f"Background median = {np.median(bg) * pixel_size_nm:.0f} nm")
    axes[1, 1].set_xlabel("Distance from PML surface (nm)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("D. Distance from PML surface")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--tomogram", required=True, type=Path,
                   help="Input tomogram (.mrc).")
    p.add_argument("--segmentation", required=True, type=Path,
                   help="PML body segmentation (.tif). Multi-label allowed; "
                        "the value specified by --pml-label is treated as PML.")
    p.add_argument("--pml-label", type=int, default=2,
                   help="Integer label that identifies the PML body in the "
                        "segmentation file (default: 2).")
    p.add_argument("--z-range", type=int, nargs=2, metavar=("Z0", "Z1"),
                   help="Inclusive Z-slice range to analyse. "
                        "If omitted, the full stack is used.")
    p.add_argument("--pixel-size-nm", type=float, required=True,
                   help="Pixel size at the working binning, in nm.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for analysis outputs.  Created if absent.")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes for per-slice segmentation "
                        "(default: 1).")
    p.add_argument("--grid-n", type=int, default=DEFAULTS["grid_n"],
                   help=f"Tile grid size (default {DEFAULTS['grid_n']}).")
    p.add_argument("--bright-pct", type=float, default=DEFAULTS["bright_pct"])
    p.add_argument("--thresh-pct", type=float, default=DEFAULTS["thresh_pct"])
    p.add_argument("--dense-pct", type=float, default=DEFAULTS["dense_pct"])
    # Edge-artefact masking: each flag gives the number of pixels to mask
    # from the corresponding edge in every slice.  May be combined.  The
    # older --artifact-x / --artifact-y names are kept as aliases for the
    # two most common cases.
    p.add_argument("--mask-left", "--artifact-x", dest="mask_left",
                   type=int, default=None,
                   help="Mask the leftmost N pixels (X < N) in every slice.")
    p.add_argument("--mask-right", dest="mask_right",
                   type=int, default=None,
                   help="Mask the rightmost N pixels (X >= width - N) in every slice.")
    p.add_argument("--mask-top", "--artifact-y", dest="mask_top",
                   type=int, default=None,
                   help="Mask the topmost N pixels (Y < N) in every slice.")
    p.add_argument("--mask-bottom", dest="mask_bottom",
                   type=int, default=None,
                   help="Mask the bottommost N pixels (Y >= height - N) in every slice.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tomogram: {args.tomogram}", flush=True)
    with mrcfile.open(args.tomogram, mode="r", permissive=True) as mrc:
        full_volume = mrc.data

        if args.z_range is not None:
            z0, z1 = args.z_range
            volume = full_volume[z0:z1 + 1].copy().astype(np.float32)
        else:
            z0, z1 = 0, full_volume.shape[0] - 1
            volume = full_volume[:].copy().astype(np.float32)

    print(f"  volume shape = {volume.shape}", flush=True)

    print(f"Loading segmentation: {args.segmentation}", flush=True)
    seg = tifffile.imread(args.segmentation)
    if seg.shape != full_volume.shape:
        sys.exit(
            f"ERROR: segmentation shape {seg.shape} does not match "
            f"tomogram shape {full_volume.shape}"
        )
    pml_mask_full = (seg == args.pml_label)
    pml_mask = pml_mask_full[z0:z1 + 1]
    if not pml_mask.any():
        sys.exit(
            f"ERROR: no voxels labelled {args.pml_label} found in the "
            "requested Z-range."
        )
    print(f"  PML voxels in range = {pml_mask.sum():,}", flush=True)

    artifact_mask = None
    edges = {
        "left":   args.mask_left,
        "right":  args.mask_right,
        "top":    args.mask_top,
        "bottom": args.mask_bottom,
    }
    if any(v is not None for v in edges.values()):
        _, ny, nx = volume.shape
        artifact_mask = np.zeros((ny, nx), dtype=bool)
        if args.mask_left is not None:
            artifact_mask[:, :args.mask_left] = True
        if args.mask_right is not None:
            artifact_mask[:, nx - args.mask_right:] = True
        if args.mask_top is not None:
            artifact_mask[:args.mask_top, :] = True
        if args.mask_bottom is not None:
            artifact_mask[ny - args.mask_bottom:, :] = True
        active = ", ".join(f"{k}={v}" for k, v in edges.items() if v is not None)
        print(f"  edge mask: {active}", flush=True)

    print(
        f"Running skeleton pipeline ({args.workers} worker"
        f"{'s' if args.workers != 1 else ''})...", flush=True
    )
    skel_stack, branch_stack = run_skeleton_volume(
        volume,
        workers=args.workers,
        bright_pct=args.bright_pct,
        thresh_pct=args.thresh_pct,
    )

    print("Building grid...", flush=True)
    grid = build_grid_connectivity(skel_stack, branch_stack, grid_n=args.grid_n)
    pml_grid = build_pml_volume_grid(pml_mask, grid["gh"], grid["gw"], args.grid_n)

    pearson_r = float(np.corrcoef(grid["composite"].ravel(), pml_grid.ravel())[0, 1])

    print("Computing dense-region distance analysis...", flush=True)
    distances = dense_distance_analysis(
        volume, pml_mask, artifact_mask=artifact_mask,
    )

    out = args.output_dir
    np.savetxt(out / "connectivity_grid.csv", grid["composite"], delimiter=",", fmt="%.6f")
    np.savetxt(out / "fiber_density_grid.csv", grid["fiber_norm"], delimiter=",", fmt="%.6f")
    np.savetxt(out / "branch_density_grid.csv", grid["branch_norm"], delimiter=",", fmt="%.6f")
    np.savetxt(out / "pml_volume_fraction_grid.csv", pml_grid, delimiter=",", fmt="%.6f")
    np.savetxt(out / "distances_dense_px.csv", distances["dense"], delimiter=",", fmt="%.3f")
    np.savetxt(out / "distances_background_px.csv", distances["background"], delimiter=",", fmt="%.3f")

    summary = {
        "tomogram": str(args.tomogram),
        "segmentation": str(args.segmentation),
        "pml_label": args.pml_label,
        "z_range": [z0, z1],
        "pixel_size_nm": args.pixel_size_nm,
        "workers": args.workers,
        "grid_n": args.grid_n,
        "bright_pct": args.bright_pct,
        "thresh_pct": args.thresh_pct,
        "dense_pct": args.dense_pct,
        "edge_mask_pixels": {k: v for k, v in edges.items() if v is not None},
        "pearson_r_connectivity_vs_pml_fraction": pearson_r,
        "mean_connectivity_pml_cells": float(
            grid["composite"][pml_grid > 0].mean()
        ) if (pml_grid > 0).any() else None,
        "mean_connectivity_non_pml_cells": float(
            grid["composite"][pml_grid == 0].mean()
        ) if (pml_grid == 0).any() else None,
        "dense_distance_median_px": float(np.median(distances["dense"]))
            if len(distances["dense"]) else None,
        "background_distance_median_px": float(np.median(distances["background"]))
            if len(distances["background"]) else None,
        "dense_distance_median_nm": float(
            np.median(distances["dense"]) * args.pixel_size_nm
        ) if len(distances["dense"]) else None,
        "background_distance_median_nm": float(
            np.median(distances["background"]) * args.pixel_size_nm
        ) if len(distances["background"]) else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Writing diagnostic figure...", flush=True)
    plot_diagnostic(
        volume, pml_mask, grid, pml_grid, distances,
        pearson_r=pearson_r,
        pixel_size_nm=args.pixel_size_nm,
        out_path=out / "diagnostic_figure.png",
    )

    print(f"\nDone.  Outputs in {out.resolve()}", flush=True)
    print(f"  Pearson r = {pearson_r:.3f}", flush=True)
    if summary["dense_distance_median_nm"] is not None:
        print(
            f"  Dense median = {summary['dense_distance_median_nm']:.0f} nm, "
            f"bg median = {summary['background_distance_median_nm']:.0f} nm",
            flush=True,
        )


if __name__ == "__main__":
    main()
