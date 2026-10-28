# pml-body-network-analysis

Quantitative analysis of the internal fibrous network of PML (promyelocytic
leukaemia) nuclear bodies from cryo-electron tomograms, and of the spatial
exclusion of high-contrast dense regions from the PML body interior.

The pipeline accompanies:

> Pražák V, Harley I, Falckenhayn J, Boutell C, Thomason PA, Davis BG,
> Kaufmann R, Carter SD. *In situ architecture of PML nuclear bodies reveals a
> porous, chromatin-permissive interior.* (submitted to Nature Cell Biology, 2026)

The base script (`pml_network_analysis.py`) is the general network pipeline.
The analyses reported in the paper (band-passed dense-cluster detection,
fibre skeleton density after cluster masking, boundary profiles and shift
tests) are in `paper_pipeline/`, described below.

## What the pipeline does

For a single tomogram plus a paired PML body segmentation, the script
`pml_network_analysis.py`:

1. Normalises each Z-slice independently (z-score, clip to +/-3 standard
   deviations, linear rescale to [0, 1]).
2. Denoises with total-variation Chambolle filtering (`weight = 0.05`).
3. Builds a bright-region exclusion mask from a Gaussian-smoothed intensity
   map (`sigma = 15`, top 25 percent of pixels excluded).
4. Enhances local contrast with CLAHE (`clip_limit = 0.02`, `kernel = 64`).
5. Extracts dark fine features with a black top-hat (`disk(6)`).
6. Thresholds at the 80th percentile of the unmasked pixels and cleans the
   binary mask (`remove_small_objects(10)`, `binary_dilation(disk(2))`,
   `remove_small_holes(50)`).
7. Skeletonises (Zhang-Suen) and identifies branch points (skeleton pixels
   with at least three neighbours).
8. Quantifies the network on a 12x12 tile grid, combining normalised fibre
   density (60 percent) and branch-point density (40 percent) into a single
   composite connectivity score.
9. Computes the Pearson correlation between per-tile connectivity and
   per-tile PML volume fraction.
10. Identifies dense pixels (bottom 15 percent of a Gaussian-smoothed slice)
    and computes the fraction of pixels classified as dense inside versus
    outside the PML body, pooled across slices (an area-normalised depletion
    measure). This simple threshold is superseded in the paper by the
    band-passed, noise-referenced cluster detection in `paper_pipeline/`.
11. Measures the unbiased pore-size distribution of the skeletonised network:
    per-slice distance transform of the inverted skeleton, local maxima give
    inscribed-circle radii of the gaps between fibres. Reports the median,
    IQR and percentages of pores large enough to admit complexes of biological
    interest (>=11 nm nucleosome, >=17 nm TRiC, >=20 nm PA28-capped
    proteasome).

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `mrcfile`, `tifffile`,
`matplotlib`.

## Usage

```bash
python pml_network_analysis.py \
    --tomogram path/to/tomogram.mrc \
    --segmentation path/to/pml_labels.tif \
    --pml-label 2 \
    --pixel-size-nm 1.0 \
    --output-dir results/ \
    --workers 16
```

### Required arguments

| Flag | Description |
|------|-------------|
| `--tomogram` | Path to the tomogram (`.mrc`). |
| `--segmentation` | Path to the PML segmentation (`.tif`). Must have the same shape as the tomogram. |
| `--output-dir` | Directory where outputs are written. Created if absent. |

### Common optional arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--pml-label` | `2` | Integer label marking the PML body in the segmentation. |
| `--z-range Z0 Z1` | full stack | Inclusive Z-range to analyse. |
| `--pixel-size-nm` | required | Pixel size at the working binning, in nm. |
| `--workers` | `1` | Parallel worker processes for per-slice segmentation. |
| `--grid-n` | `12` | Tile-grid size for connectivity analysis. |
| `--bright-pct` | `75` | Percentile above which intensity is treated as bright background. |
| `--thresh-pct` | `80` | Percentile for binarising the black top-hat response. |
| `--dense-pct` | `15` | Bottom intensity percentile defining "dense" regions. |
| `--mask-left` | off | Mask the leftmost N pixels in every slice (reconstruction edge artefact). Alias: `--artifact-x`. |
| `--mask-right` | off | Mask the rightmost N pixels in every slice. |
| `--mask-top` | off | Mask the topmost N pixels in every slice. Alias: `--artifact-y`. |
| `--mask-bottom` | off | Mask the bottommost N pixels in every slice. Any combination of edge flags may be used. |

Edge masks are applied only to the dense-region analysis (so that
edge-reconstruction artefacts are not mis-classified as biological dense
regions). The skeleton and pore-size analyses run on the full volume because
the masked columns may sit inside the PML body and still contain real
fibrous network on top of the artefact.

Run `python pml_network_analysis.py --help` for the full list.

## Input requirements

- **Tomogram**: `.mrc`, float or integer data, at the working binning.
  The reference settings in the paper use bin 4 (pixel size 9.92 A).
- **Segmentation**: `.tif` of identical shape to the tomogram. Multi-label
  segmentations are supported; only the value given to `--pml-label` is
  treated as PML body.

## Output files

All outputs are written into `--output-dir`:

| File | Contents |
|------|----------|
| `summary.json` | All parameters, Pearson `r`, mean connectivity inside/outside the PML body, inside/outside dense fraction and fold-depletion, pore-size statistics. |
| `connectivity_grid.csv` | grid_n x grid_n composite connectivity score. |
| `fiber_density_grid.csv` | Normalised fibre density per tile. |
| `branch_density_grid.csv` | Normalised branch-point density per tile. |
| `pml_volume_fraction_grid.csv` | PML volume fraction per tile. |
| `pore_diameters_nm.csv` | All measured pore diameters in nanometres. |
| `diagnostic_figure.png` | Four-panel QC figure: connectivity overlay, PML volume fraction, connectivity vs PML volume fraction scatter, dense fraction inside vs outside. |
| `pore_size_histogram.png` | Pore-diameter histogram with reference lines at nucleosome (11 nm), TRiC (17 nm) and PA28-capped proteasome (20 nm) diameters. |

## Importable API

All stages are exposed as functions and can be called from another Python
program:

```python
from pml_network_analysis import (
    segment_slice,
    run_skeleton_volume,
    build_grid_connectivity,
    build_pml_volume_grid,
    dense_mask_slice,
    dense_fraction_inside_outside,
)
```

## Licence

MIT (see `LICENSE`).

## Pipeline used for the submitted manuscript (`paper_pipeline/`, October 2026)

The scripts in `paper_pipeline/` are the exact code behind Fig. 2 and
Supplementary Fig. 1 of the submitted manuscript. They import the slice
normalisation, denoising, CLAHE, black top-hat, thresholding and skeletonisation
functions from `pml_network_analysis.py` unchanged, and add the steps described
in the Methods:

| Script | Purpose |
|---|---|
| `fibre_after_blob_removal.py` | Per-slice fibre skeleton density inside and outside the segmented body after masking dense clusters of 15 nm and larger (patch filled with local background plus matched noise). Masks a 60 px reconstruction-artefact strip at the right edge of every tomogram (`T_ALL`). Run with `isonet` or `raw`. |
| `controls_fibre_denominator_bandpass.py` | Band-passed dense-cluster counts by size class (source of Fig. 2c and of the raw versus IsoNet comparison in Supplementary Fig. 1g) and denominator controls for the fibre ratio. |
| `boundary_profile.py`, `boundary_profile_rm15.py` | Dense coverage, large-cluster density and skeleton density in 10 nm shells from -120 to +150 nm around the body boundary, and the 20 nm erosion/dilation test (20 iterations of a cross-shaped structuring element, ratios as means over slices of per-slice fractions). |
| `boundary_shift_sweep.py` | Boundary shift sweep from 10 to 50 nm (only the 20 nm result is reported). |
| `body_vs_body_fibre.py` | Per-slice interior fibre density compared between the five bodies (exploratory; the test is not reported in the paper). |
| `fig2bc_panels.py`, `fig2_assemble_with_render.py` | Fig. 2 panels and assembly. |
| `fig_supp2_assemble_600dpi.py` | Supplementary Fig. 1 assembly (the file name is from an earlier draft numbering). |
| `rerun_rightmask_2026-10-25.sh` | The order in which the scripts were run for the submitted numbers. |

Paths to the tomograms, segmentations and output folders are hard-coded at the
top of each script and must be edited for another machine. Tomogram and
segmentation files are not part of this repository. Tilt-series and tomograms will be
deposited in EMPIAR, and the PML body segmentations are available from the
corresponding author. The Fig. 2a ChimeraX render is read as a PNG and is not
included.

Note: the pore-size measurement (step 11 above) and the composite connectivity
score are not used in the submitted manuscript, which reports skeleton density,
strand length and dense-cluster depletion only.
