import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
IMG_TOP, IMG_BOT = 9811, 18958          # colorbar & plot image vertical extent (SVG units)
cbar_major = {                           # (y of 10^0 tick, y of highest decade tick, that decade)
    "Liverpool":   (18833, 10185, 4),
    "Orlando":     (18833, 10278, 5),
    "Rockefeller": (18833, 10247, 4),
}
# time axis: y major ticks 10..35 min; m/z axis: x major ticks 600..1400
ty = np.array([18958, 17153, 15318, 13513, 11678, 9873]); tv = np.arange(10, 36, 5)
mx = np.array([3744, 5424, 7072, 8751, 10399, 12078, 13726, 15405, 17053]); mv = np.arange(600, 1401, 100)
X0, XW, Y0, YH = 3682, 13464, 9811, 9147
W, H = 433, 294
pix_x = X0 + (np.arange(W) + 0.5) * XW / W
pix_y = Y0 + (np.arange(H) + 0.5) * YH / H
mz = np.polyval(np.polyfit(mx, mv, 1), pix_x)
rt = np.polyval(np.polyfit(ty, tv, 1), pix_y)

out = {}
for s, (y0, yk, k) in cbar_major.items():
    per_dec = (y0 - yk) / k
    logy = lambda y: (y0 - y) / per_dec
    lo, hi = logy(IMG_BOT), logy(IMG_TOP)
    cbar = np.load(f"{s}_cbar.npy").astype(float)           # 256 rows, top -> bottom
    lv = hi - (np.arange(256) + 0.5) / 256 * (hi - lo)       # log10 intensity per level
    img = np.asarray(Image.open(f"{s}_bitmap.png").convert("RGB")).astype(float)
    d, idx = cKDTree(cbar).query(img.reshape(-1, 3))
    inten = 10 ** lv[idx]
    inten[(img.reshape(-1, 3) == 255).all(1)] = 0.0          # pure white = below scale / no signal
    inten = inten.reshape(H, W)
    exact = (d == 0).mean() * 100
    print(f"{s}: colorbar 10^{lo:.2f}–10^{hi:.2f}, exact color matches {exact:.1f}%, max color dist {d.max():.1f}, "
          f"white {np.mean(inten==0)*100:.1f}%, total intensity {inten.sum():.3g}")
    out[s] = inten
np.savez("decoded.npz", mz=mz, rt=rt, **out)
print("m/z", mz[0].round(1), "→", mz[-1].round(1), "| RT", rt[0].round(2), "→", rt[-1].round(2))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

d = np.load("decoded.npz")
mz, rt = d["mz"], d["rt"]
strains = ["Liverpool", "Orlando", "Rockefeller"]
colors = {"Liverpool": "#00BCD4", "Orlando": "#7B3FA0", "Rockefeller": "#F28C28"}
X = np.vstack([d[s].ravel() for s in strains])          # 3 samples x 127,302 m/z-RT pixels, raw
pca = PCA(n_components=2)
sc = pca.fit_transform(X)
var = pca.explained_variance_ratio_ * 100
print("variance %", var.round(1))
print(pd.DataFrame(sc, index=strains, columns=["PC1", "PC2"]).round(0))

fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.15], wspace=0.45)
ax = fig.add_subplot(gs[0])
ax.axhline(0, color="#bbb", lw=0.8, zorder=0); ax.axvline(0, color="#bbb", lw=0.8, zorder=0)
for i, s in enumerate(strains):
    ax.scatter(sc[i, 0], sc[i, 1], s=180, color=colors[s], edgecolor="#333", lw=0.8, zorder=3, label=s)
    ax.annotate(s, (sc[i, 0], sc[i, 1]), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9)
ax.set_xlabel(f"PC1 ({var[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("Scores", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
pad = lambda v: (v.min() - 0.25 * np.ptp(v), v.max() + 0.25 * np.ptp(v))
ax.set_xlim(*pad(sc[:, 0])); ax.set_ylim(*pad(sc[:, 1]))

ext = [mz[0], mz[-1], rt[-1], rt[0]]
for j in range(2):
    a = fig.add_subplot(gs[j + 1])
    L = pca.components_[j].reshape(len(rt), len(mz))
    lim = np.percentile(np.abs(L), 99.9)
    im = a.imshow(L, extent=ext, aspect="auto", cmap="RdBu_r",
                  norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    a.set_xlabel("m/z"); a.set_ylabel("Retention time (min)")
    hi = " & ".join(st for st, v in zip(strains, sc[:, j]) if v > 0.25 * np.abs(sc[:, j]).max())
    lo = " & ".join(st for st, v in zip(strains, sc[:, j]) if v < -0.25 * np.abs(sc[:, j]).max())
    a.set_title(f"PC{j+1} loadings\nred = higher in {hi}; blue = higher in {lo}", fontsize=9.5)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02); cb.set_label("Loading", fontsize=8)
    cb.ax.tick_params(labelsize=7)
fig.suptitle("Raw PCA of top-down MS1 maps (unscaled intensities, m/z × RT pixels as variables)", fontsize=12, y=1.0)
fig.savefig("topdown_raw_pca.png", dpi=200, bbox_inches="tight")
fig.savefig("topdown_raw_pca.pdf", bbox_inches="tight")
pd.DataFrame(sc, index=strains, columns=["PC1", "PC2"]).assign(
    **{"PC1 % var": var[0], "PC2 % var": var[1]}).to_csv("topdown_raw_pca_scores.csv")
