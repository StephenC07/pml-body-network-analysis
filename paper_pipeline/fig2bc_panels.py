#!/usr/bin/env python3
"""Fig. 2b: dense clusters per um^2 inside vs outside by size class, five bodies (band-passed
detection, every 3rd slice through the body).  Fig. 2c: dense coverage and >20 nm clusters vs
distance from the CLEM boundary, five bodies + mean.  Same style as Fig. 2a."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = [sys.argv[0], "isonet"]
sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted")
import fibre_after_blob_removal as F
PIX = F.PIX; R = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/results_v2/"; OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/figures/"
C2 = json.load(open(R + "controls_fibre_denominator_bandpass_isonet.json")); BP = json.load(open(R + "boundary_profile.json"))
BODIES = list(F.T); CLASSES = ["6 to 15", "15 to 20", "20 to 30", "> 30"]; GROUPS = [(0, 2), (2, 3), (3, 5), (5, 8)]
MK = dict(TS_01="o", TS_03="s", TS_08="^", TS_15="D", TS_16="v")
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})

# analysed areas (um^2, every 3rd slice) from the labels
area = {}
for t, (_, segp, (z0, z1), mr) in F.T.items():
    pml = F.seg_load(segp); art = np.zeros((1024, 1024), bool); art[:, :100] = True
    if mr: art[:, 1024 - mr:] = True
    zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; zs = list(range(zidx[0], zidx[-1] + 1, 3))
    area[t] = ((pml[zs] & ~art).sum() * PIX ** 2 / 1e6, ((~pml[zs]) & ~art).sum() * PIX ** 2 / 1e6)

dens = {t: {} for t in BODIES}
for t in BODIES:
    ni = np.array(C2[t]["control2"]["bandpass"]["n_in"]); no = np.array(C2[t]["control2"]["bandpass"]["n_out"])
    for c, (a, b) in zip(CLASSES, GROUPS):
        dens[t][c] = (ni[a:b].sum() / area[t][0], no[a:b].sum() / area[t][1])

# ---------- 2b
fig, ax = plt.subplots(figsize=(3.4, 2.9))
x = np.arange(4); w = 0.36
mi = [np.mean([dens[t][c][0] for t in BODIES]) for c in CLASSES]; mo = [np.mean([dens[t][c][1] for t in BODIES]) for c in CLASSES]
ax.bar(x - w / 2, mi, w, color="#c7d6ea", edgecolor="#1f4e79", lw=0.8, label="inside PML body")
ax.bar(x + w / 2, mo, w, color="#f3c6c6", edgecolor="#8b1a1a", lw=0.8, label="nucleoplasm")
for t in BODIES:
    ax.plot(x - w / 2, [dens[t][c][0] for c in CLASSES], MK[t], color="#1f4e79", ms=3.5, mfc="white", mew=0.8)
    ax.plot(x + w / 2, [dens[t][c][1] for c in CLASSES], MK[t], color="#8b1a1a", ms=3.5, mfc="white", mew=0.8)
for i, c in enumerate(CLASSES):
    r = mo[i] / mi[i]; ax.text(x[i], max(mo[i], mi[i]) * 1.35, f"{r:.1f}x", ha="center", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(1, 3000); ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_xlabel("dense cluster size (nm)")
ax.set_ylabel("clusters per µm²"); ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.spines[["top", "right"]].set_visible(False); ax.text(-0.22, 1.02, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT + "Fig2b_clusters_by_size.png", dpi=300); fig.savefig(OUT + "Fig2b_clusters_by_size.pdf"); plt.close(fig)
ratios = {c: (mo[i] / mi[i], [dens[t][c][1] / dens[t][c][0] for t in BODIES]) for i, c in enumerate(CLASSES)}

# ---------- 2c
edges = np.array(BP["TS_01"]["edges"]); mid = (edges[:-1] + edges[1:]) / 2; sel = (mid >= -110) & (mid <= 140)
fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
for a, key, yl, mult in ((ax[0], "cov", "dense coverage (%)", 100), (ax[1], "big", "clusters > 20 nm per µm²", 1)):
    Y = []
    for t in BODIES:
        y = np.array(BP[t][key]) * mult; Y.append(y); a.plot(mid[sel], y[sel], "-", color="#888888", lw=0.7, alpha=0.8)
    a.plot(mid[sel], np.mean(Y, axis=0)[sel], "-", color="#1f4e79" if key == "cov" else "#8b1a1a", lw=2.0, label="mean of five bodies")
    a.axvspan(-110, 0, color="#dce6f2", alpha=0.6, lw=0); a.axvline(0, color="k", lw=0.9)
    a.set_ylabel(yl); a.spines[["top", "right"]].set_visible(False); a.set_xlim(-110, 140)
ax[0].text(-100, ax[0].get_ylim()[1] * 0.97, "PML body", fontsize=7, va="top"); ax[0].text(8, ax[0].get_ylim()[1] * 0.97, "nucleoplasm", fontsize=7, va="top")
ax[0].legend(frameon=False, fontsize=7, loc="lower right"); ax[1].set_xlabel("distance from PML body boundary (nm)")
ax[0].text(-0.22, 1.05, "c", transform=ax[0].transAxes, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT + "Fig2c_boundary_profile.png", dpi=300); fig.savefig(OUT + "Fig2c_boundary_profile.pdf"); plt.close(fig)

json.dump(dict(area_um2=area, density_per_um2=dens, ratio_out_in=ratios), open(OUT + "Fig2b_data.json", "w"), indent=1)
print("2b ratios outside/inside (mean of five; per body):")
for c in CLASSES:
    print(f"  {c:8s} nm  {ratios[c][0]:.2f}   " + " ".join(f"{v:.2f}" for v in ratios[c][1]))
