#!/usr/bin/env python3
"""
Raw-feature PCA across three LC-MS1 2-D heatmaps (.xyz), built for multi-GB files.

Each file is streamed in chunks (never fully loaded) and binned onto ONE shared
x/y grid (e.g. retention time x m/z). Each grid bin is a sample; the three
heatmaps' intensities in that bin are its three features. PCA is then run on
those bins.

Run from Anaconda Prompt:
    python heatmap_pca_stream.py ORL.xyz LIV.xyz ROCK.xyz --names Orlando Liverpool Rockefeller

Useful options:
    --bins 1000 1000        grid size (x bins, y bins); default 1000 x 1000
    --xrange 0 120          skip the range-finding pass if you know the limits
    --yrange 300 2000
    --agg mean|sum|max      how points falling in one bin are combined (default mean)
    --no-log                PCA on raw intensity (default: log10(1 + intensity))
    --no-standardize        covariance PCA instead of correlation PCA
    --drop-empty            ignore bins that are zero in all three maps (default on)
    --out pca_out           output folder

The binned grids are cached in <out>/binned.npz, so re-running with different
PCA options (--no-log, --no-standardize) skips re-reading the big files, as long
as --bins/--xrange/--yrange/--agg stay the same.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------- reading -------
def sniff(path, max_check=200):
    """Return (n_header_lines, separator)."""
    skip = 0
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_check:
                break
            s = line.strip()
            tok = s.replace(",", " ").replace(";", " ").replace("\t", " ").split()
            ok = False
            if len(tok) >= 3:
                try:
                    [float(t) for t in tok[:3]]
                    ok = True
                except ValueError:
                    pass
            if ok:
                if "," in s:
                    return skip, ","
                if ";" in s:
                    return skip, ";"
                if "\t" in s:
                    return skip, "\t"
                return skip, r"\s+"
            skip += 1
    sys.exit(f"Could not find numeric x y z rows in the first {max_check} lines of {path}")


def chunks(path, chunksize, cols=(0, 1, 2)):
    skip, sep = sniff(path)
    kw = dict(sep=sep, skiprows=skip, header=None, usecols=list(cols),
              dtype=np.float64, comment="#", chunksize=chunksize, engine="c")
    # skip malformed rows; the keyword differs between pandas versions
    major, minor = (int(v) for v in pd.__version__.split(".")[:2])
    if (major, minor) >= (1, 3):
        kw["on_bad_lines"] = "skip"
    else:
        kw["error_bad_lines"] = False
        kw["warn_bad_lines"] = False
    reader = pd.read_csv(path, **kw)
    for ch in reader:
        a = ch.to_numpy()
        a = a[np.isfinite(a).all(axis=1)]
        yield a


def human(n):
    for u in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{u}"
        n /= 1000
    return f"{n:.1f}T"


def find_ranges(files, chunksize):
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for f in files:
        t = time.time(); n = 0
        print(f"  range pass: {os.path.basename(f)}", flush=True)
        for a in chunks(f, chunksize, cols=(0, 1)):
            if len(a) == 0:
                continue
            xmin = min(xmin, a[:, 0].min()); xmax = max(xmax, a[:, 0].max())
            ymin = min(ymin, a[:, 1].min()); ymax = max(ymax, a[:, 1].max())
            n += len(a)
            print(f"\r    {human(n)} rows  ({time.time()-t:.0f}s)", end="", flush=True)
        print()
    return (xmin, xmax), (ymin, ymax)


def bin_file(path, xr, yr, nx, ny, agg, chunksize):
    tot = np.zeros(nx * ny, dtype=np.float64)
    cnt = np.zeros(nx * ny, dtype=np.int64)
    mx = np.full(nx * ny, -np.inf) if agg == "max" else None
    t = time.time(); n = 0
    print(f"  binning: {os.path.basename(path)}", flush=True)
    for a in chunks(path, chunksize):
        if len(a) == 0:
            continue
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        keep = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
        x, y, z = x[keep], y[keep], z[keep]
        ix = np.minimum(((x - xr[0]) / (xr[1] - xr[0]) * nx).astype(np.int64), nx - 1)
        iy = np.minimum(((y - yr[0]) / (yr[1] - yr[0]) * ny).astype(np.int64), ny - 1)
        idx = iy * nx + ix
        tot += np.bincount(idx, weights=z, minlength=nx * ny)
        cnt += np.bincount(idx, minlength=nx * ny)
        if mx is not None:
            np.maximum.at(mx, idx, z)
        n += len(a)
        print(f"\r    {human(n)} rows  ({time.time()-t:.0f}s)", end="", flush=True)
    print()
    if agg == "sum":
        g = tot
    elif agg == "max":
        g = np.where(np.isfinite(mx), mx, 0.0)
    else:
        g = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    return g.reshape(ny, nx), cnt.reshape(ny, nx)


# ----------------------------------------------------------------- PCA -------
def pca(X, standardize=True):
    mu = X.mean(axis=0)
    Xc = X - mu
    if standardize:
        sd = Xc.std(axis=0, ddof=1)
        Xc = Xc / np.where(sd == 0, 1, sd)
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    evals, evecs = np.linalg.eigh(C)
    o = np.argsort(evals)[::-1]
    evals, evecs = evals[o], evecs[:, o]
    for j in range(evecs.shape[1]):
        if evecs[np.argmax(np.abs(evecs[:, j])), j] < 0:
            evecs[:, j] *= -1
    scores = Xc @ evecs
    loadings = evecs * np.sqrt(np.maximum(evals, 0))
    return evals, evecs, loadings, scores


# --------------------------------------------------------------- plots -------
def plots(grids, names, mask, evals, evecs, loadings, scores, xr, yr, a):
    out = a.out
    pcs = ["PC1", "PC2", "PC3"]
    ratio = evals / evals.sum()
    ext = [xr[0], xr[1], yr[0], yr[1]]
    cols = ["#4C72B0", "#55A868", "#C44E52"]
    zlab = "log10(1+I)" if not a.no_log else "intensity"

    # scree
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(pcs, ratio * 100, color="#4C72B0")
    ax.plot(pcs, np.cumsum(ratio) * 100, "o-", color="#C44E52", label="cumulative")
    for i, r in enumerate(ratio):
        ax.text(i, r * 100 + 1.5, f"{r:.1%}", ha="center")
    ax.set_ylim(0, 108); ax.set_ylabel("Explained variance (%)"); ax.legend()
    ax.set_title("Scree")
    fig.tight_layout(); fig.savefig(f"{out}/scree.png", dpi=200); plt.close(fig)

    # loadings
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for j, ax in enumerate(axes):
        ax.bar(names, loadings[:, j], color=cols)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{pcs[j]} ({ratio[j]:.1%})")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Loading" + ("" if a.no_standardize else " (correlation)"))
    fig.tight_layout(); fig.savefig(f"{out}/loadings.png", dpi=200); plt.close(fig)

    # biplot
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hb = ax.hexbin(scores[:, 0], scores[:, 1], gridsize=150, bins="log",
                   cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="bins (log10 count)")
    s = 0.8 * np.percentile(np.abs(scores[:, :2]), 99.9) / max(np.abs(loadings[:, :2]).max(), 1e-12)
    for i, n in enumerate(names):
        ax.annotate("", xy=(loadings[i, 0] * s, loadings[i, 1] * s), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text(loadings[i, 0] * s * 1.12, loadings[i, 1] * s * 1.12, n,
                color="red", fontweight="bold", ha="center")
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})"); ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.set_title("Biplot (each point = one grid bin)")
    fig.tight_layout(); fig.savefig(f"{out}/biplot.png", dpi=200); plt.close(fig)

    # inputs + score maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, n in enumerate(names):
        im = axes[0, j].imshow(grids[j], origin="lower", aspect="auto", cmap="magma",
                               extent=ext, interpolation="nearest")
        axes[0, j].set_title(f"{n} ({zlab})"); fig.colorbar(im, ax=axes[0, j])
    for j in range(3):
        img = np.full(mask.shape, np.nan)
        img[mask] = scores[:, j]
        lim = np.nanpercentile(np.abs(img), 99.5)
        im = axes[1, j].imshow(img, origin="lower", aspect="auto", cmap="RdBu_r",
                               vmin=-lim, vmax=lim, extent=ext, interpolation="nearest")
        axes[1, j].set_title(f"{pcs[j]} score ({ratio[j]:.1%})")
        fig.colorbar(im, ax=axes[1, j])
    for ax in axes.ravel():
        ax.set_xlabel(a.xlabel); ax.set_ylabel(a.ylabel)
    fig.tight_layout(); fig.savefig(f"{out}/score_maps.png", dpi=200); plt.close(fig)

    # difference-from-consensus maps are what PC2/PC3 usually encode; also show correlation
    X = np.stack([g[mask] for g in grids], axis=1)
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=20)
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax); ax.set_title("Pearson r across bins")
    fig.tight_layout(); fig.savefig(f"{out}/correlation.png", dpi=200); plt.close(fig)

    pd.DataFrame({"PC": pcs, "eigenvalue": evals, "explained_var": ratio,
                  "cumulative": np.cumsum(ratio)}).to_csv(f"{out}/variance.csv", index=False)
    pd.DataFrame(loadings, index=names, columns=pcs).to_csv(f"{out}/loadings.csv")
    pd.DataFrame(evecs, index=names, columns=pcs).to_csv(f"{out}/eigenvectors.csv")
    pd.DataFrame(corr, index=names, columns=names).to_csv(f"{out}/correlation.csv")

    ny, nx = mask.shape
    xc = xr[0] + (np.arange(nx) + 0.5) * (xr[1] - xr[0]) / nx
    yc = yr[0] + (np.arange(ny) + 0.5) * (yr[1] - yr[0]) / ny
    YY, XX = np.meshgrid(yc, xc, indexing="ij")
    df = pd.DataFrame({"x": XX[mask], "y": YY[mask],
                       **{n: X[:, i] for i, n in enumerate(names)},
                       **{p: scores[:, i] for i, p in enumerate(pcs)}})
    df.to_csv(f"{out}/bin_scores.csv.gz", index=False, compression="gzip", float_format="%.6g")

    print("\nExplained variance:")
    for p, r in zip(pcs, ratio):
        print(f"  {p}: {r:.2%}")
    print("\nLoadings:")
    print(pd.DataFrame(loadings, index=names, columns=pcs).round(3).to_string())
    print("\nCorrelation:")
    print(pd.DataFrame(corr, index=names, columns=names).round(3).to_string())
    print(f"\nOutputs in: {os.path.abspath(out)}")


# ---------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs=3)
    ap.add_argument("--names", nargs=3)
    ap.add_argument("--bins", nargs=2, type=int, default=[1000, 1000], metavar=("NX", "NY"))
    ap.add_argument("--xrange", nargs=2, type=float)
    ap.add_argument("--yrange", nargs=2, type=float)
    ap.add_argument("--agg", choices=["mean", "sum", "max"], default="mean")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep bins that are zero in all three maps")
    ap.add_argument("--xlabel", default="x (column 1)")
    ap.add_argument("--ylabel", default="y (column 2)")
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    ap.add_argument("--out", default="pca_out")
    a = ap.parse_args()

    names = a.names or [os.path.splitext(os.path.basename(f))[0] for f in a.files]
    nx, ny = a.bins
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "binned.npz")
    key = f"{[os.path.abspath(f) for f in a.files]}|{nx}|{ny}|{a.xrange}|{a.yrange}|{a.agg}"

    grids = None
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=False)
        if str(c["key"]) == key:
            print(f"Using cached bins from {cache}")
            grids = [c["g0"], c["g1"], c["g2"]]
            xr, yr = tuple(c["xr"]), tuple(c["yr"])
    if grids is None:
        t0 = time.time()
        if a.xrange and a.yrange:
            xr, yr = tuple(a.xrange), tuple(a.yrange)
        else:
            print("Pass 1/2: finding the shared x/y range...")
            fxr, fyr = find_ranges(a.files, a.chunksize)
            xr = tuple(a.xrange) if a.xrange else fxr
            yr = tuple(a.yrange) if a.yrange else fyr
        print(f"Grid: x {xr[0]:g}-{xr[1]:g}, y {yr[0]:g}-{yr[1]:g}, {nx} x {ny} bins")
        print("Pass 2/2: binning...")
        grids = []
        for f in a.files:
            g, cnt = bin_file(f, xr, yr, nx, ny, a.agg, a.chunksize)
            print(f"    filled bins: {(cnt > 0).mean():.1%}")
            grids.append(g)
        np.savez_compressed(cache, key=np.array(key), g0=grids[0], g1=grids[1],
                            g2=grids[2], xr=np.array(xr), yr=np.array(yr))
        print(f"Binning done in {(time.time()-t0)/60:.1f} min (cached to {cache})")

    raw = [np.clip(g, 0, None) for g in grids] if not a.no_log else grids
    feats = [np.log10(1 + g) for g in raw] if not a.no_log else raw
    mask = np.ones_like(feats[0], dtype=bool) if a.keep_empty else \
        ~np.all(np.stack([g == 0 for g in grids]), axis=0)
    X = np.stack([f[mask] for f in feats], axis=1)
    print(f"\nPCA on {len(X):,} bins x 3 heatmaps "
          f"({'log10(1+I)' if not a.no_log else 'raw I'}, "
          f"{'covariance' if a.no_standardize else 'correlation'} PCA)")
    evals, evecs, loadings, scores = pca(X, standardize=not a.no_standardize)
    plots(feats, names, mask, evals, evecs, loadings, scores, xr, yr, a)


if __name__ == "__main__":
    main()
