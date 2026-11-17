# PCA of LC-MS 2-D heatmaps from *Aedes aegypti* strains

Python scripts used to compare LC-MS maps of histones from the Orlando, Liverpool and
Rockefeller mosquito strains by principal component analysis (PCA), as described in:

> [Andrea Ramirez Torres1, Md Shofiul Alam1, Cassandra N. Fuller2,3, Kevin Jeanne Dit Fouque1, Lilian Valadares Tose3, Richard M. Searfoss4, Francisca N. de Luna Vitorino4, Mariangela Kosmopoulou5, Detlev Suckau5, Benjamin A. Garcia4, and Francisco Fernandez-Lima1*]. [Histone Landscape Differences Across Ae. aegypti Strains]. *[Journal Pending]* ([2026]). [DOI when available]



| Script | What it does | Used for |
|---|---|---|
| `heatmap_pca_stream.py` | Bins three LC-MS1 heatmaps onto a shared RT × m/z grid and runs a **bin-level PCA** (each grid bin is an observation; each strain is a variable) | Figure [X]b, c, e, f |
| `strain_pca.py` | Runs a **strain-level PCA** on the binned grids from `heatmap_pca_stream.py` (each strain is a sample; each bin is a variable) | Figure [X]d |
| `replicate_pca.py` | **Replicate-level PCA** of bottom-up LC-MS runs, with RT × m/z and (optionally) ion-mobility × m/z maps; includes clustering, correlation and a PERMANOVA test | Figure [Y] |

---

## Requirements

- Python ≥ 3.8 (tested with Python [version])
- numpy, pandas, scipy, matplotlib

```
pip install numpy pandas scipy matplotlib
```

Versions used for the manuscript: numpy [x.y.z], pandas [x.y.z], scipy [x.y.z], matplotlib [x.y.z].

The scripts run on Windows, macOS and Linux. They read input files in chunks, so they handle
multi-gigabyte files without loading them fully into memory.

## Input format

Each input is a plain-text `.xyz` file exported from [instrument software, version], one row per
data point:

```
x  m/z  intensity
```

- `x` is retention time in seconds (RT maps) or ion mobility (mobility maps).
- Columns may be separated by spaces, tabs, commas or semicolons; header lines are skipped
  automatically.
- In `replicate_pca.py`, the strain name is taken from the text before the first `_` in the
  file name, and the run number from the last number of three or more digits
  (e.g. `Orlando_..._1440.xyz` → strain *Orlando*, run *1440*).

---

## 1. Bin-level PCA — `heatmap_pca_stream.py`

Bins three heatmaps onto a common grid, log-transforms the intensities and runs a correlation
PCA in which each RT × m/z bin is an observation and the three strains are the variables.

**Command used for the manuscript:**

```
python heatmap_pca_stream.py Orlando.xyz Liverpool.xyz Rockefeller.xyz ^
    --names Orlando Liverpool Rockefeller ^
    --xrange 600 2100 --yrange 400 1400 ^
    --xlabel "RT (s)" --ylabel "m/z" --out pca_cropped
```

(`^` continues a line in the Windows command prompt; use `\` on macOS/Linux.)

**Processing:** intensities are averaged within each of 1,000 × 1,000 bins → log₁₀(1 + I) →
bins empty in all strains are removed → each strain is standardised to unit variance → PCA by
eigendecomposition of the 3 × 3 correlation matrix.

**Main options**

| Option | Default | Meaning |
|---|---|---|
| `--bins NX NY` | `1000 1000` | Grid size |
| `--xrange MIN MAX`, `--yrange MIN MAX` | full data range | Window; giving both skips the range-finding pass |
| `--agg mean\|sum\|max` | `mean` | How points within a bin are combined |
| `--no-log` | off | Use raw instead of log₁₀(1 + I) intensities |
| `--no-standardize` | off | Covariance instead of correlation PCA |
| `--keep-empty` | off | Keep bins that are empty in all strains |

**Outputs** (in `--out`): `score_maps.png` (input maps and PC score maps), `biplot.png`,
`loadings.png`, `scree.png`, `correlation.png`, and the tables `variance.csv`, `loadings.csv`,
`eigenvectors.csv`, `correlation.csv`, `bin_scores.csv.gz`. The binned grids are cached in
`binned.npz`; re-running with different PCA options reuses them.

## 2. Strain-level PCA — `strain_pca.py`

Treats each strain as one sample and every bin as a variable, using the grids cached by
`heatmap_pca_stream.py`. With three samples there are at most two components, so PC1 + PC2
always explain 100% of the variance; the plot shows the relative distances between strains.

**Command used for the manuscript:**

```
python strain_pca.py pca_cropped/binned.npz --names Orlando Liverpool Rockefeller --out pca_cropped
```

Sensitivity check for differences in total signal between runs:

```
python strain_pca.py pca_cropped/binned.npz --norm median --tag strain_pca_mediannorm --out pca_cropped
```

**Processing:** log₁₀(1 + I) → optional per-strain median normalisation (`--norm median`) →
mean-centring of each bin across strains → PCA by singular value decomposition.
`--scale pareto|auto` adds per-bin scaling (default: none).

**Outputs:** `strain_pca.png` / `.pdf`, `strain_pca_scores.csv`, `strain_pca_variance.csv`,
`strain_pca_distances.csv` (Euclidean distances between strains).

## 3. Replicate-level PCA — `replicate_pca.py`

Each `.xyz` file is one LC-MS run. Runs are summed onto a shared grid, normalised and compared
by PCA. Mobility × m/z maps can be added; runs are matched to their RT files by strain and run
number, and the script then reports RT-only, mobility-only and combined analyses.

**Command used for the manuscript:**

```
python replicate_pca.py --rt-folder "RT vs. mz" --mobility-folder "Mobility vs. mz" ^
    --rt [MIN MAX] --mz [MIN MAX] --out replicate_pca
```

**Processing:** points within the m/z window are kept (the upper edge is exclusive, which also
removes the per-scan padding points at the m/z limits) → intensities summed into RT × m/z bins
(10 s × 1 m/z by default) → total-ion-current (TIC) normalisation → log₁₀(1 + I) → bins with
signal in at least two runs kept → mean-centring → PCA by SVD. In the combined analysis, each
data block is divided by the square root of its total variance so that RT and mobility data
contribute equally.

Group separation is tested by PERMANOVA on the Euclidean distances between runs, using every
distinct relabelling of runs into groups. With two replicates of each of three strains there are
only 15 distinct relabellings, so the smallest attainable p-value is 1/15 ≈ 0.067.

**Main options**

| Option | Default | Meaning |
|---|---|---|
| `--rt-folder PATH` | — | Folder of RT × m/z `.xyz` files (**set this**; the built-in default is the authors' local path) |
| `--mobility-folder PATH` | none | Folder of mobility × m/z `.xyz` files |
| `--rt MIN MAX` | `300 3900` | RT window (s) |
| `--mz MIN MAX` | `200 1700` | m/z window |
| `--rt-bin`, `--mz-bin` | `10`, `1` | Bin widths |
| `--mob auto\|MIN MAX`, `--mob-bins N` | full range, one bin per mobility step | Mobility window and bins |
| `--norm tic\|median\|none` | `tic` | Per-run normalisation |
| `--scale none\|pareto\|auto` | `none` | Per-bin scaling |
| `--min-samples N` | `2` | Minimum number of runs with signal for a bin to be kept |
| `--rename A=B ...` | `LVP=Liverpool ORL=Orlando ROCK=Rockefeller` | Map file-name prefixes to strain names |
| `--exclude WORDS` | none | Skip files whose names contain these words |

**Outputs** (in `--out`, or in `rt_mz/`, `mobility_mz/` and `combined/` subfolders when mobility
data are included): `pca_PC1_PC2.png` / `.pdf`, `pca_overview.png` (score plots, scree plot,
dendrogram, correlation matrix, replicate-vs-strain distances), `pca_loading_maps.png`, and the
tables `scores.csv`, `variance.csv`, `distances.csv`, `correlation.csv`, `top_loading_bins.csv`,
`runs.csv`. Per-file m/z profiles are cached in `_cache/`, so later runs with different binning
or normalisation do not re-read the raw files (changing `--mz` or `--mz-bin` does).

---

## Data

The raw LC-MS data are available from [repository, e.g. PRIDE] under accession [PXD000000].

## Citation

If you use these scripts, please cite the article above and the archived code:

> [Authors] ([year]). *[Title of Zenodo record]* (Version 1.0) [Software]. Zenodo. [DOI]

## Licence

Released under the [MIT] licence; see `LICENSE`.

## Contact

[Andrea Ramirez Torres], [University of Georgia] — [asr18942@uga.edu]
