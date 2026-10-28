#!/usr/bin/env python3
"""Assemble Figure 2 with the ChimeraX isosurface render of TS_03 as panel a. 180 x 225 mm.
Output Figure2_with_render.png/.pdf. The render PNG is not part of this repository.
a: ChimeraX render of the TS_03 body with mapped particles
b: TS_03 central slice (rotated ROT degrees to match Fig. 1) + dense clusters by size
c: clusters per um^2 by size class, five bodies
d: dense coverage and >20 nm clusters vs distance from the boundary
e: fibre skeleton crops inside / nucleoplasm
f: fibre skeleton per area (bar) + fibre profile across the boundary
usage: fig2_assemble.py [ROT degrees, default 180]
"""
import json, sys
import numpy as np
import mrcfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy import ndimage as ndi
ROT = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
sys.argv = [sys.argv[0], "isonet"]
sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted")
import fibre_after_blob_removal as F
pna = F.pna; PIX = F.PIX; K, SIG = 1.0, 5.0
R = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/results_v2/"; OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/figures/"
FB = json.load(open(R + "fibre_after_blob_removal_isonet.json")); BP = json.load(open(R + "boundary_profile.json")); B2 = json.load(open(OUT + "Fig2b_data.json"))
import os
if os.path.exists(R + "boundary_profile_rm15.json"):
    BP15 = json.load(open(R + "boundary_profile_rm15.json")); print("fibre profile: blobs >= 15 nm removed")
else:
    BP15 = BP; print("WARNING: boundary_profile_rm15.json not found; fibre profile still shows the nothing-removed skeleton")
BODIES = list(F.T); MK = dict(TS_01="o", TS_03="s", TS_08="^", TS_15="D", TS_16="v"); BLUE, RED = "#1f4e79", "#8b1a1a"
CLASSES = ["6 to 15", "15 to 20", "20 to 30", "> 30"]; SIZE_EDGES = [6, 15, 20, 30, 1000]; COLS = ["#f2c14e", "#1b9e77", "#e7298a", "#1f1f8f"]
cmap = ListedColormap(COLS); norm = BoundaryNorm(SIZE_EDGES, cmap.N)
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.7, "axes.titlesize": 7.5, "legend.fontsize": 6.5})
MAIN = "TS_03"; W = 300


def noise_thr(x):
    h, e = np.histogram(x, bins=512); h = ndi.gaussian_filter1d(h.astype(float), 3)
    m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]); sd = np.sqrt(np.mean((x[x > m] - m) ** 2)); return m - K * sd


def rot(a):
    if ROT % 360 == 0: return a
    if ROT % 360 == 180: return a[::-1, ::-1]
    return ndi.rotate(a, ROT, reshape=False, order=0 if a.dtype == bool or a.dtype.kind in "iu" else 1, cval=0)


def crop_box(region, w=W, skel=None):
    """box fully inside region; if skel given, the box whose skeleton density is closest to the region mean (representative)."""
    dt = ndi.distance_transform_edt(region); h = w // 2
    if skel is None:
        cy, cx = np.unravel_index(dt.argmax(), dt.shape); cy = int(np.clip(cy, h, 1023 - h)); cx = int(np.clip(cx, h, 1023 - h)); return slice(cy - h, cy + h), slice(cx - h, cx + h)
    target = skel[region].mean(); best, bd = None, 1e9
    for cy in range(h, 1024 - h, 25):
        for cx in range(h + 100, 1024 - h, 25):
            if dt[cy, cx] >= h * 1.05:
                d = abs(skel[cy - h:cy + h, cx - h:cx + h].mean() - target)
                if d < bd: best, bd = (slice(cy - h, cy + h), slice(cx - h, cx + h)), d
    return best


def letter(ax, s, dx=-0.08, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def paired_bar(a, vi, vo, ylabel, fmt):
    w = 0.36
    a.bar(-w / 2, np.mean(vi), w, color="#c7d6ea", edgecolor=BLUE, lw=0.7); a.bar(w / 2, np.mean(vo), w, color="#f3c6c6", edgecolor=RED, lw=0.7)
    for t, i, o in zip(BODIES, vi, vo):
        a.plot([-w / 2, w / 2], [i, o], "-", color="#999999", lw=0.5); a.plot(-w / 2, i, MK[t], color=BLUE, ms=3, mfc="white", mew=0.7); a.plot(w / 2, o, MK[t], color=RED, ms=3, mfc="white", mew=0.7)
    a.text(0, max(vi) * 1.06, fmt.format(np.mean(vi) / np.mean(vo)), ha="center", fontsize=6.5)
    a.set_xticks([-w / 2, w / 2]); a.set_xticklabels(["PML\nbody", "nucleo-\nplasm"]); a.set_ylabel(ylabel); a.set_ylim(0, max(vi) * 1.2); a.spines[["top", "right"]].set_visible(False)


def profile(a, key, ylabel, mult, color, label_body=True, src=None):
    src = BP if src is None else src
    edges = np.array(src["TS_01"]["edges"]); mid = (edges[:-1] + edges[1:]) / 2; sel = (mid >= -110) & (mid <= 140); Y = []
    for t in BODIES:
        y = np.array(src[t][key]) * mult; Y.append(y); a.plot(mid[sel], y[sel], "-", color="#888888", lw=0.6, alpha=0.8)
    a.plot(mid[sel], np.mean(Y, axis=0)[sel], "-", color=color, lw=1.8)
    a.axvspan(-110, 0, color="#dce6f2", alpha=0.6, lw=0); a.axvline(0, color="k", lw=0.8); a.set_xlim(-110, 140); a.set_ylabel(ylabel); a.spines[["top", "right"]].set_visible(False)
    if label_body:
        a.text(-104, a.get_ylim()[1] * 0.99, "PML body", fontsize=6.5, va="top"); a.text(6, a.get_ylim()[1] * 0.99, "nucleoplasm", fontsize=6.5, va="top")


# ---------- data for a and d (TS_03 central slice)
mrc, segp, (z0, z1), mr = F.T[MAIN]; pml = F.seg_load(segp); art = np.zeros((1024, 1024), bool); art[:, :100] = True
if mr: art[:, 1024 - mr:] = True
zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; z = int(zidx[len(zidx) // 2])
sl = mrcfile.open(mrc, permissive=True).data[z].astype(np.float32); s = (sl - sl.mean()) / (sl.std() + 1e-12)
bp = ndi.gaussian_filter(s, SIG) - ndi.gaussian_filter(s, 40.0); mask = (bp < noise_thr(bp[~art])) & ~art
m = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2)); lab, n = ndi.label(m)
area = np.bincount(lab.ravel()); d = 2 * np.sqrt(area / np.pi) * PIX; d[0] = 0; d[d < 6] = 0; size = d[lab]
rng15 = np.random.default_rng(int(z)); sl15 = F.remove(sl, F.blob_diam_map(sl, art), 15, rng15)
skel, _ = pna.segment_slice(sl15); skel &= ~art   # panel e: blobs >= 15 nm masked before skeletonisation (same as panel f)

# ---------- figure: 180 mm x 165 mm, 3 rows x 3 columns, consistent cells
fig = plt.figure(figsize=(180 / 25.4, 195 / 25.4))
# row 0: a render | b slice + clusters ; row 1: c bar | d profiles ; row 2: e crops | f bar + profile
gsT = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.08, left=0.095, right=0.985, top=0.965, bottom=0.715)
gsB = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.5, left=0.095, right=0.985, top=0.60, bottom=0.385)
gsB2 = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.55], wspace=0.28, left=0.095, right=0.985, top=0.285, bottom=0.055)
axR = fig.add_subplot(gsT[0]); axA1 = fig.add_subplot(gsT[1]); axA2 = fig.add_subplot(gsT[2])
axB = fig.add_subplot(gsB[0]); axC1 = fig.add_subplot(gsB[1]); axC2 = fig.add_subplot(gsB[2], sharex=axC1)
gsD = gsB2[0].subgridspec(1, 2, wspace=0.08); axD1 = fig.add_subplot(gsD[0]); axD2 = fig.add_subplot(gsD[1])
gsE = gsB2[1].subgridspec(1, 2, wspace=0.5, width_ratios=[0.68, 1.2]); axE1 = fig.add_subplot(gsE[0]); axE2 = fig.add_subplot(gsE[1])

# render panel (ChimeraX, TS_03 PML body isosurface with mapped-back TRiC and PA28)
RENDER = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/Nature_Cell_Biology/figure5_new/chimerax/render_fig2_cropped.png"
axR.imshow(plt.imread(RENDER)); axR.set_anchor('N'); axR.set_xticks([]); axR.set_yticks([])
for sp in axR.spines.values(): sp.set_visible(False)
LEG_R = ([Patch(facecolor="#F3E6C4", edgecolor="k", lw=0.5), Patch(color="#e0191c"), Patch(color="#87cefa"), Patch(color="#ffd700")], ["PML body", "TRiC closed", "TRiC open", "PA28 proteasome"])

# a
x0, x1 = 100, 1024 - (mr or 0); img_a = rot(s[:, x0:x1]); pml_a = rot(pml[z][:, x0:x1]); size_a = rot(size[:, x0:x1])
for a in (axA1, axA2):
    a.imshow(img_a, cmap="gray", vmin=-2.5, vmax=2.5, interpolation="bilinear"); a.contour(pml_a, levels=[0.5], colors="white", linewidths=1.3)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
axA2.imshow(np.ma.masked_where(size_a == 0, size_a), cmap=cmap, norm=norm, interpolation="nearest", alpha=0.95)
H, Wd = img_a.shape; axA1.plot([15, 15 + 150 / PIX], [H - 25, H - 25], "w-", lw=2.5); axA1.text(15, H - 40, "150 nm", color="w", fontsize=6.5, fontweight="bold")
ys, xs = np.nonzero(pml_a); axA1.text(xs.mean(), ys.mean(), "PML body", color="white", fontsize=7.5, ha="center", fontweight="bold")
axA2.set_title("dense clusters by size")
LEG_A = ([Patch(color=c) for c in COLS], [f"{c} nm" for c in CLASSES])

# b
dens = B2["density_per_um2"]; x = np.arange(4); w = 0.36
mi = [np.mean([dens[t][c][0] for t in BODIES]) for c in CLASSES]; mo = [np.mean([dens[t][c][1] for t in BODIES]) for c in CLASSES]
axB.bar(x - w / 2, mi, w, color="#c7d6ea", edgecolor=BLUE, lw=0.7, label="PML body"); axB.bar(x + w / 2, mo, w, color="#f3c6c6", edgecolor=RED, lw=0.7, label="nucleoplasm")
for t in BODIES:
    axB.plot(x - w / 2, [dens[t][c][0] for c in CLASSES], MK[t], color=BLUE, ms=3, mfc="white", mew=0.7); axB.plot(x + w / 2, [dens[t][c][1] for c in CLASSES], MK[t], color=RED, ms=3, mfc="white", mew=0.7)
for i in range(4):
    axB.text(x[i], max(mo[i], mi[i]) * 1.4, f"{mi[i]/mo[i]:.1f}x", ha="center", fontsize=6.5)
axB.set_yscale("log"); axB.set_ylim(1, 4000); axB.set_xticks(x); axB.set_xticklabels(["6 to 15", "15 to 20", "20 to 30", "> 30"], fontsize=6.5, rotation=20); axB.set_xlabel("dense cluster size (nm)"); axB.set_ylabel("clusters per µm²")
axB.legend(frameon=False, loc="upper right"); axB.spines[["top", "right"]].set_visible(False)

# c
profile(axC1, "cov", "dense coverage (%)", 100, BLUE); profile(axC2, "big", "clusters > 20 nm per µm²", 1, BLUE, label_body=False)
axC1.set_xlabel("distance from boundary (nm)"); axC2.set_xlabel("distance from boundary (nm)")

# d
for a, (name, reg) in zip((axD1, axD2), (("inside PML body", pml[z] & ~art), ("nucleoplasm", ~pml[z] & ~art))):
    ys, xs = crop_box(reg, skel=skel); s15 = (sl15 - sl15.mean()) / (sl15.std() + 1e-12); a.imshow(s15[ys, xs], cmap="gray", vmin=-2.5, vmax=2.5, interpolation="bicubic")
    sk = ndi.binary_dilation(skel[ys, xs], iterations=1).astype(float); sk = ndi.gaussian_filter(sk, 0.6)
    a.contourf(sk, levels=[0.5, 2.0], colors=["#ffa500"], alpha=0.9, antialiased=True)   # vector skeleton outline
    a.set_xticks([]); a.set_yticks([]); a.set_title(name)
    for sp in a.spines.values(): sp.set_visible(False)
    a.text(0.97, 0.04, f"fibre {100*skel[ys, xs].mean():.1f}%", transform=a.transAxes, color="w", fontsize=6.5, ha="right")
axD1.plot([12, 12 + 50 / PIX], [W - 12, W - 12], "w-", lw=2.2); axD1.text(12, W - 20, "50 nm", color="w", fontsize=6.5, fontweight="bold")

# e
FVAR = "3 >=15 nm removed"   # headline fibre variant (Stephen, 28 Sept 2026): blobs >= 15 nm masked, 6 to 15 nm clusters kept
si = [100 * FB[t][FVAR]["skel_in"] for t in BODIES]; so = [100 * FB[t][FVAR]["skel_out"] for t in BODIES]
paired_bar(axE1, si, so, "fibre skeleton (% of area)", "{:.2f}x"); axE1.set_xticklabels(["PML\nbody", "nucleo-\nplasm"], fontsize=6)
profile(axE2, "skel", "fibre skeleton (% of area)", 100, BLUE, src=BP15); axE2.set_xlabel("distance from boundary (nm)")

# consistent panel letters (figure coordinates) and the a legend
fig.canvas.draw()
for ax, L in ((axR, "a"), (axA1, "b"), (axB, "c"), (axC1, "d"), (axD1, "e"), (axE1, "f")):
    b = ax.get_position(); x_ = b.x0 - 0.045
    fig.text(x_, b.y1 + 0.004, L, fontsize=11, fontweight="bold", va="bottom", ha="left")
bA1, bA2 = axA1.get_position(), axA2.get_position()
fig.legend(*LEG_A, loc="upper center", bbox_to_anchor=((bA1.x0 + bA2.x1) / 2, bA1.y0 - 0.005), ncol=2, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=1.0)
bRr = axR.get_position(); fig.legend(*LEG_R, loc="upper center", bbox_to_anchor=((bRr.x0 + bRr.x1) / 2, bRr.y0 - 0.005), ncol=2, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
fig.savefig(OUT + "Figure2_with_render.png", dpi=600); fig.savefig(OUT + "Figure2_with_render.pdf", dpi=600)
print("written", ROT)
