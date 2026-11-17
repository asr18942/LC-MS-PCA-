#!/usr/bin/env python3
"""
Strain-level PCA: each strain's LC-MS1 map is one sample, each RT x m/z bin is a feature.

Reads the binned grids cached by heatmap_pca_stream.py (binned.npz), so it runs in seconds.

    python strain_pca.py pca_cropped\\binned.npz --names Orlando Liverpool Rockefeller --out pca_cropped

Preprocessing (in order):
  1. log10(1 + intensity)
  2. drop bins that are empty in all strains
  3. optional per-strain median normalisation (--norm median) to remove overall
     brightness differences between runs
  4. mean-centre each bin across strains (standard PCA); --scale pareto|auto adds scaling

With 3 samples there are at most 2 non-trivial PCs, so PC1 + PC2 = 100% of the variance.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--names", nargs="+", default=["Orlando", "Liverpool", "Rockefeller"])
    ap.add_argument("--norm", choices=["none", "median"], default="none")
    ap.add_argument("--scale", choices=["none", "pareto", "auto"], default="none")
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="strain_pca")
    a = ap.parse_args()

    c = np.load(a.npz, allow_pickle=False)
    grids = [c[k] for k in sorted([k for k in c.files if k.startswith("g")],
                                  key=lambda s: int(s[1:]))]
    names = a.names[:len(grids)]
    raw = np.stack([np.clip(g, 0, None).ravel() for g in grids])          # strains x bins
    keep = ~np.all(raw == 0, axis=0)
    X = np.log10(1 + raw[:, keep])

    if a.norm == "median":
        X = X - np.median(X, axis=1, keepdims=True) + np.median(X)

    Xc = X - X.mean(axis=0)
    if a.scale != "none":
        sd = X.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1, sd)
        Xc = Xc / (np.sqrt(sd) if a.scale == "pareto" else sd)

    # SVD on the small strains x bins matrix
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (Xc.shape[0] - 1)
    ratio = var / var.sum()
    scores = U * S
    for j in range(scores.shape[1]):          # deterministic signs
        if Vt[j, np.argmax(np.abs(Vt[j]))] < 0:
            scores[:, j] *= -1
            Vt[j] *= -1

    # pairwise Euclidean distances (what the plot geometry reflects)
    D = np.sqrt(((Xc[:, None, :] - Xc[None, :, :]) ** 2).sum(-1))

    os.makedirs(a.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for i, n in enumerate(names):
        ax.scatter(scores[i, 0], scores[i, 1], s=220, color=COLORS[i % len(COLORS)],
                   edgecolor="black", linewidth=1, zorder=3, label=n)
        ax.annotate(n, (scores[i, 0], scores[i, 1]), xytext=(10, 8),
                    textcoords="offset points", fontsize=12, fontweight="bold")
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color="grey", lw=0.8, ls="--", zorder=1)
    lim = np.abs(scores[:, :2]).max() * 1.35
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})", fontsize=12)
    sub = "log10(1+I), mean-centred"
    if a.norm == "median":
        sub += ", median-normalised"
    if a.scale != "none":
        sub += f", {a.scale}-scaled"
    ax.set_title(f"Strain PCA of LC-MS1 maps\n{keep.sum():,} RT×m/z bins ({sub})",
                 fontsize=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    base = os.path.join(a.out, a.tag)
    fig.savefig(base + ".png", dpi=300); fig.savefig(base + ".pdf"); plt.close(fig)

    pd.DataFrame(scores[:, :2], index=names, columns=["PC1", "PC2"]).to_csv(base + "_scores.csv")
    pd.DataFrame({"PC": ["PC1", "PC2"], "explained_var": ratio[:2]}).to_csv(
        base + "_variance.csv", index=False)
    pd.DataFrame(D, index=names, columns=names).to_csv(base + "_distances.csv")

    print(pd.DataFrame(scores[:, :2], index=names, columns=["PC1", "PC2"]).round(2))
    print(f"PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%}  ({keep.sum():,} bins)")
    print("Distances:\n", pd.DataFrame(D, index=names, columns=names).round(1))
    print(f"Saved {base}.png / .pdf")


if __name__ == "__main__":
    main()
