#!/usr/bin/env python3
"""Supplementary Fig. 1 (file name from an earlier draft numbering), supporting Fig. 2:
a-d  the other four bodies (TS_01, TS_08, TS_15, TS_16): central slice + dense clusters by size
e-g  fibre strand length, boundary-shift test, and cluster ratio (nucleoplasm / body) per size
     class on IsoNet and on raw tomograms
"""
import json, sys
import numpy as np
import mrcfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy import ndimage as ndi
sys.argv = [sys.argv[0], "isonet"]
sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted")
import fibre_after_blob_removal as F
PIX = F.PIX; K, SIG = 1.0, 5.0
R = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/results_v2/"; OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/figures/"
FB = json.load(open(R + "fibre_after_blob_removal_isonet.json"))
C2 = {src: json.load(open(R + f"controls_fibre_denominator_bandpass_{src}.json")) for src in ("isonet", "raw")}
BODIES = list(F.T); OTHERS = ["TS_01", "TS_08", "TS_15", "TS_16"]; MK = dict(TS_01="o", TS_03="s", TS_08="^", TS_15="D", TS_16="v"); BLUE, RED = "#1f4e79", "#8b1a1a"
CLASSES = ["6 to 15", "15 to 20", "20 to 30", "> 30"]; GROUPS = [(0, 2), (2, 3), (3, 5), (5, 8)]; SIZE_EDGES = [6, 15, 20, 30, 1000]; COLS = ["#f2c14e", "#1b9e77", "#e7298a", "#1f1f8f"]
cmap = ListedColormap(COLS); norm = BoundaryNorm(SIZE_EDGES, cmap.N)
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.7, "axes.titlesize": 7.5, "legend.fontsize": 6.5})


def noise_thr(x):
    h, e = np.histogram(x, bins=512); h = ndi.gaussian_filter1d(h.astype(float), 3)
    m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]); sd = np.sqrt(np.mean((x[x > m] - m) ** 2)); return m - K * sd


def slice_data(t):
    mrc, segp, (z0, z1), mr = F.T[t]; pml = F.seg_load(segp); art = np.zeros((1024, 1024), bool); art[:, :100] = True
    if mr: art[:, 1024 - mr:] = True
    zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; z = int(zidx[len(zidx) // 2])
    sl = mrcfile.open(mrc, permissive=True).data[z].astype(np.float32); s = (sl - sl.mean()) / (sl.std() + 1e-12)
    bp = ndi.gaussian_filter(s, SIG) - ndi.gaussian_filter(s, 40.0); mask = (bp < noise_thr(bp[~art])) & ~art
    m = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2)); lab, n = ndi.label(m)
    area = np.bincount(lab.ravel()); d = 2 * np.sqrt(area / np.pi) * PIX; d[0] = 0; d[d < 6] = 0
    x1 = 1024 - (60 if mr else 0); sl_ = (slice(0, 1024), slice(100, x1))
    return s[sl_], d[lab][sl_], pml[z][sl_], z


fig = plt.figure(figsize=(180 / 25.4, 180 / 25.4))
gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 1.25], hspace=0.12, wspace=0.08, left=0.045, right=0.98, top=0.965, bottom=0.05)
axes_img = []
for i, t in enumerate(OTHERS):
    a1 = fig.add_subplot(gs[i, 0]); a2 = fig.add_subplot(gs[i, 1]); axes_img.append(a1)
    img, size, pml, z = slice_data(t)
    for a in (a1, a2):
        a.imshow(img, cmap="gray", vmin=-2.5, vmax=2.5, interpolation="bilinear"); a.contour(pml, levels=[0.5], colors="white", linewidths=1.2)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    a2.imshow(np.ma.masked_where(size == 0, size), cmap=cmap, norm=norm, interpolation="nearest", alpha=0.95)
    H = img.shape[0]; a1.plot([15, 15 + 150 / PIX], [H - 25, H - 25], "w-", lw=2.2); a1.text(15, H - 42, "150 nm", color="w", fontsize=6.5, fontweight="bold")
    ys, xs = np.nonzero(pml); a1.text(xs.mean(), ys.mean(), "PML body", color="white", fontsize=7, ha="center", fontweight="bold")
    a1.text(0.03, 0.97, t, transform=a1.transAxes, color="w", fontsize=7.5, fontweight="bold", va="top")
    if i == 0: a1.set_title("cryo-tomogram slice"); a2.set_title("dense clusters by size")

# right column
gsR = gs[:, 2].subgridspec(3, 1, hspace=0.55, height_ratios=[1.0, 1.0, 1.0])
gsE = gsR[0].subgridspec(1, 2, width_ratios=[1, 1.1], wspace=0.6); axE1 = fig.add_subplot(gsE[0]); axG = fig.add_subplot(gsE[1])
axF = fig.add_subplot(gsR[1]); axLeg = fig.add_subplot(gsR[2]); axLeg.axis("off")

# e: connected strand length at native fibre density (all blobs removed)
li = [FB[t]["4 all blobs removed"]["conn_in"]["seg_len_nm"] for t in BODIES]; lo = [FB[t]["4 all blobs removed"]["conn_out"]["seg_len_nm"] for t in BODIES]; w = 0.36
axE1.bar(-w / 2, np.mean(li), w, color="#c7d6ea", edgecolor=BLUE, lw=0.7); axE1.bar(w / 2, np.mean(lo), w, color="#f3c6c6", edgecolor=RED, lw=0.7)
for t, i, o in zip(BODIES, li, lo):
    axE1.plot([-w / 2, w / 2], [i, o], "-", color="#999999", lw=0.5); axE1.plot(-w / 2, i, MK[t], color=BLUE, ms=3, mfc="white", mew=0.7); axE1.plot(w / 2, o, MK[t], color=RED, ms=3, mfc="white", mew=0.7)
axE1.text(0, max(li) * 1.06, f"{np.mean(li)/np.mean(lo):.1f}x", ha="center", fontsize=6.5)
axE1.set_xticks([-w / 2, w / 2]); axE1.set_xticklabels(["PML\nbody", "nucleo-\nplasm"], fontsize=6.5); axE1.set_ylabel("connected strand length (nm)"); axE1.set_ylim(0, max(li) * 1.2)
axE1.spines[["top", "right"]].set_visible(False)

# f: robustness, ratio nucleoplasm / body per size class, IsoNet vs raw (same labels, every 3rd slice)
B2 = json.load(open(OUT + "Fig2b_data.json")); AREA = B2["area_um2"]
x = np.arange(4)
for src, col, lab, off in (("isonet", BLUE, "IsoNet denoised", -0.17), ("raw", "#888888", "raw reconstruction", 0.17)):
    vals = []
    for t in BODIES:
        ni = np.array(C2[src][t]["control2"]["bandpass"]["n_in"]); no = np.array(C2[src][t]["control2"]["bandpass"]["n_out"])
        vals.append([(no[a:b].sum() / AREA[t][1]) / max(ni[a:b].sum() / AREA[t][0], 1e-9) for a, b in GROUPS])
    vals = np.array(vals)
    di = np.array([[ni_[a:b].sum() / AREA[t][0] for a, b in GROUPS] for t, ni_ in zip(BODIES, [np.array(C2[src][t]["control2"]["bandpass"]["n_in"]) for t in BODIES])])
    do = np.array([[no_[a:b].sum() / AREA[t][1] for a, b in GROUPS] for t, no_ in zip(BODIES, [np.array(C2[src][t]["control2"]["bandpass"]["n_out"]) for t in BODIES])])
    m = do.mean(axis=0) / di.mean(axis=0)   # ratio of five-body means, as in Fig. 2c
    axF.bar(x + off, m, 0.32, color="white", edgecolor=col, lw=0.9, label=lab)
    for k, t in enumerate(BODIES):
        axF.plot(x + off, vals[k], MK[t], color=col, ms=3, mfc="white", mew=0.7, ls="none")
axF.axhline(1, color="k", ls="--", lw=0.7); axF.set_yscale("log"); axF.set_ylim(0.3, 40); axF.set_xticks(x); axF.set_xticklabels(CLASSES, fontsize=6.5, rotation=20)
axF.set_xlabel("dense cluster size (nm)"); axF.set_ylabel("clusters, nucleoplasm / body"); axF.legend(frameon=False, loc="upper left"); axF.spines[["top", "right"]].set_visible(False)

# g: boundary position sensitivity, inside / outside ratio at the label, 20 nm eroded and 20 nm dilated
BP = json.load(open(R + "boundary_profile.json")); BP15 = json.load(open(R + "boundary_profile_rm15.json"))
GM = [("cov", BP, "dense coverage", "#1f4e79"), ("big", BP, "clusters > 20 nm", "#e7298a"), ("skel", BP15, "fibre skeleton", "#d95f02")]
xs_ = [-20, 0, 20]
for key, src, lab, col in GM:
    allr = []
    for t in BODIES:
        r = [src[t][k][key][0] / src[t][k][key][1] for k in ("shifted_in", "label", "shifted_out")]; allr.append(r)
        axG.plot(xs_, r, "-", color=col, lw=0.6, alpha=0.5); axG.plot(xs_, r, MK[t], color=col, ms=3, mfc="white", mew=0.7, ls="none")
    m_ = np.mean(allr, axis=0); axG.plot(xs_, m_, "-", color=col, lw=1.8, label=lab)
axG.axhline(1, color="k", ls="--", lw=0.7); axG.set_ylim(0.2, 1.75); axG.set_yticks([0.25, 0.5, 0.75, 1, 1.25, 1.5]); axG.set_yticklabels(["0.25", "0.5", "", "1", "", "1.5"])
axG.set_xticks(xs_); axG.set_xticklabels(["\u221220", "0", "+20"]); axG.set_xlim(-26, 26); axG.set_xlabel("boundary shift (nm)"); axG.set_ylabel("inside / outside ratio", labelpad=1)
axG.legend(frameon=False, fontsize=5.5, loc="center", bbox_to_anchor=(0.5, 0.62), handlelength=1.2, labelspacing=0.2, borderaxespad=0); axG.spines[["top", "right"]].set_visible(False)

# legends and letters
axLeg.legend([Patch(color=c) for c in COLS], [f"{c} nm" for c in CLASSES], loc="upper left", ncol=1, frameon=False, fontsize=7, title="dense cluster size", title_fontsize=7)
axLeg.legend([plt.Line2D([], [], marker=MK[t], color="#444444", ls="none", mfc="white", ms=4) for t in BODIES], BODIES, loc="upper right", frameon=False, fontsize=7, title="body", title_fontsize=7)
axLeg.add_artist(axLeg.legend([Patch(color=c) for c in COLS], [f"{c} nm" for c in CLASSES], loc="upper left", ncol=1, frameon=False, fontsize=7, title="dense cluster size", title_fontsize=7))
axLeg.legend([plt.Line2D([], [], marker=MK[t], color="#444444", ls="none", mfc="white", ms=4) for t in BODIES], BODIES, loc="upper right", frameon=False, fontsize=7, title="body", title_fontsize=7)
fig.canvas.draw()
for ax, L in list(zip(axes_img, "abcd")) + [(axE1, "e"), (axG, "f"), (axF, "g")]:
    b = ax.get_position(); fig.text(b.x0 - (0.03 if L in "abcd" else 0.065), b.y1 - (0.0 if L in "abcd" else -0.004), L, fontsize=11, fontweight="bold", va="bottom", ha="left")
fig.savefig(OUT + "SuppFig2_assembled_600dpi.png", dpi=600); fig.savefig(OUT + "SuppFig2_assembled_600dpi.pdf", dpi=600)
print("written")
