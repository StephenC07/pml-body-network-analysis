# pml-body-network-analysis

Quantitative analysis of the internal fibrous network of PML (promyelocytic
leukaemia) nuclear bodies from cryo-electron tomograms, and of the spatial
exclusion of high-contrast dense regions from the PML body interior.

The pipeline accompanies:

> Pražák V, Harley I, Boutell C, Thomason PA, Carlin LM, Davis BG, Kaufmann R,
> Carter SD. *In situ molecular architecture of eYFP-PML.I bodies by cryo-CLEM
> and cryo-electron tomography reveals trinucleosomes in an open state.*

It reproduces the per-tomogram network quantification and the dense-region
distance analysis described in the paper, without committing to any specific
dataset.

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
10. Computes the Euclidean distance from each dense pixel (bottom 15 percent
    of a Gaussian-smoothed slice) to the nearest PML body surface voxel,
    and compares the distribution to a size-matched random background sample.

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
| `--artifact-x` | off | If set, mask all pixels with `X < value` from every slice. |

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
| `summary.json` | All parameters, Pearson `r`, mean connectivity inside/outside the PML body, median dense/background distances in pixels and nm. |
| `connectivity_grid.csv` | grid_n x grid_n composite connectivity score. |
| `fiber_density_grid.csv` | Normalised fibre density per tile. |
| `branch_density_grid.csv` | Normalised branch-point density per tile. |
| `pml_volume_fraction_grid.csv` | PML volume fraction per tile. |
| `distances_dense_px.csv` | Dense-pixel distances from the PML surface, in bin-4 pixels. |
| `distances_background_px.csv` | Random background distances, in bin-4 pixels. |
| `diagnostic_figure.png` | Four-panel QC figure: connectivity overlay, PML volume fraction, connectivity vs PML volume fraction scatter, distance histogram. |

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
    dense_distance_analysis,
)
```

## Licence

MIT (see `LICENSE`).
