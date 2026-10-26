#!/usr/bin/env python3
"""VARIANT (Mac, 28 Sept 2026): fibre skeleton profile computed after removing blobs >= 15 nm (6 to 15 nm kept).
Dense coverage and cluster profiles unchanged. Output: results_v2/boundary_profile_rm15.json. No figures.
Boundary profile, five bodies (IsoNet, every 3rd slice through the body):
dense coverage (band-passed, K=1), >20 nm cluster density and fibre skeleton density as a
function of signed distance from the CLEM label boundary (negative = inside), in 10 nm shells
from -120 to +150 nm.  Also inside/outside ratios with the label eroded / dilated by 20 nm.
Output: results_v2/boundary_profile.json, figures/boundary_profile_5bodies.png,
figures/boundary_profile_panels.png (one slice per body, label + +-20 nm contours over the
band-passed dark mask and the skeleton).
"""
import json, sys
import numpy as np
import mrcfile
from multiprocessing import Pool
from scipy import ndimage as ndi
sys.argv = [sys.argv[0], "isonet"]
sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted")
import fibre_after_blob_removal as F
pna = F.pna; PIX = F.PIX; K, SIG = 1.0, 5.0
EDGES = np.arange(-120, 160, 10); OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/"


def noise_thr(x):
    h, e = np.histogram(x, bins=512); h = ndi.gaussian_filter1d(h.astype(float), 3)
    m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]); sd = np.sqrt(np.mean((x[x > m] - m) ** 2)); return m - K * sd


def work(args):
    t, z, sl, art, pml, example = args
    s = (sl - sl.mean()) / (sl.std() + 1e-12); bp = ndi.gaussian_filter(s, SIG) - ndi.gaussian_filter(s, 40.0)
    mask = (bp < noise_thr(bp[~art])) & ~art
    m = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2)); lab, n = ndi.label(m)
    big = np.zeros((1024, 1024), bool)
    if n:
        area = np.bincount(lab.ravel())[1:]; d = 2 * np.sqrt(area / np.pi) * PIX
        com = np.array(ndi.center_of_mass(m, lab, range(1, n + 1)))[d > 20]
        cy = np.clip(np.round(com[:, 0]).astype(int), 0, 1023); cx = np.clip(np.round(com[:, 1]).astype(int), 0, 1023); big[cy, cx] = True
    rng = np.random.default_rng(int(z)); img = F.remove(sl, F.blob_diam_map(sl, art), 15, rng)
    skel, _ = pna.segment_slice(img); skel &= ~art
    # signed distance (nm): negative inside the label
    dist = (ndi.distance_transform_edt(~pml) - ndi.distance_transform_edt(pml)) * PIX
    ok = ~art
    idx = np.digitize(dist, EDGES) - 1
    prof = {}
    for k, img in (("cov", mask), ("big", big), ("skel", skel)):
        num = np.bincount(idx[ok & (idx >= 0) & (idx < len(EDGES) - 1)], weights=img[ok & (idx >= 0) & (idx < len(EDGES) - 1)].astype(float), minlength=len(EDGES) - 1)
        prof[k] = num.tolist()
    prof["area"] = np.bincount(idx[ok & (idx >= 0) & (idx < len(EDGES) - 1)], minlength=len(EDGES) - 1).tolist()
    # eroded / dilated label ratios
    er = ndi.binary_erosion(pml, iterations=20); di = ndi.binary_dilation(pml, iterations=20)
    for name, ins, outs in (("label", pml, ~pml), ("eroded20", er, ~di), ("dilated20", di, ~pml & ~di | (~di)), ("shifted_in", er, ~pml), ("shifted_out", di, ~di)):
        ins = ins & ok; outs = outs & ok
        if ins.sum() < 500:
            continue
        prof[name] = dict(cov=[float(mask[ins].mean()), float(mask[outs].mean())], big=[float(big[ins].sum() / ins.sum()), float(big[outs].sum() / outs.sum())], skel=[float(skel[ins].mean()), float(skel[outs].mean())])
    if example:
        pass
    return t, prof


def main():
    tasks = []
    for t, (mrc, segp, (z0, z1), mr) in F.T.items():
        pml = F.seg_load(segp); art = np.zeros((1024, 1024), bool); art[:, :100] = True
        if mr: art[:, 1024 - mr:] = True
        zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; zs = list(range(zidx[0], zidx[-1] + 1, 3)); zc = zs[len(zs) // 2]
        with mrcfile.open(mrc, permissive=True) as m:
            tasks += [(t, z, m.data[z].astype(np.float32), art, pml[z], z == zc) for z in zs]
    with Pool(9) as pool:
        R = pool.map(work, tasks, chunksize=1)
    results = {}
    for t in F.T:
        P = [p for tt, p in R if tt == t]; area = np.sum([p["area"] for p in P], axis=0).astype(float)
        results[t] = dict(edges=EDGES.tolist(), area=area.tolist())
        for k in ("cov", "big", "skel"):
            v = np.sum([p[k] for p in P], axis=0) / np.maximum(area, 1)
            results[t][k] = (v * (1e6 / PIX ** 2) if k == "big" else v).tolist()
        for name in ("label", "shifted_in", "shifted_out"):
            Q = [p[name] for p in P if name in p]
            results[t][name] = {k: [float(np.mean([q[k][0] for q in Q])), float(np.mean([q[k][1] for q in Q]))] for k in ("cov", "big", "skel")}
        r = results[t]
        print(f"\n=== {t} ===  in/out ratios: label as drawn | label shrunk 20 nm (inside only core) | label grown 20 nm (outside starts 20 nm further)")
        for k, lab in (("cov", "dense coverage"), ("big", ">20 nm clusters"), ("skel", "skeleton density")):
            f = lambda n: r[n][k][0] / max(r[n][k][1], 1e-9)
            print(f"   {lab:18s} {f('label'):6.2f} | {f('shifted_in'):6.2f} | {f('shifted_out'):6.2f}")
        c = np.array(r["cov"]); mid = (EDGES[:-1] + EDGES[1:]) / 2
        print("   coverage profile (nm from boundary: value): " + "  ".join(f"{m:+.0f}:{100*v:.0f}%" for m, v in zip(mid, c) if -60 <= m <= 90))
    json.dump(results, open(OUT + "results_v2/boundary_profile_rm15.json", "w"), indent=1)
    return

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mid = (EDGES[:-1] + EDGES[1:]) / 2
    fig, ax = plt.subplots(3, 5, figsize=(22, 11))
    for j, t in enumerate(F.T):
        r = results[t]
        for i, (k, yl, mult) in enumerate((("cov", "dense coverage (%)", 100), ("big", "clusters > 20 nm per um2", 1), ("skel", "fibre skeleton (px per 100 px)", 100))):
            a = ax[i, j]; v = np.array(r[k]) * mult; ar = np.array(r["area"])
            a.plot(mid, v, "o-", color="#d62728" if k == "skel" else "#1f77b4", ms=4)
            a.axvspan(-120, 0, color="#f4cccc", alpha=0.35); a.axvline(0, color="k", lw=1.2); a.axvline(-20, color="k", ls=":", lw=0.8); a.axvline(20, color="k", ls=":", lw=0.8)
            a.set_xlim(-120, 150); a.set_xlabel("distance from CLEM boundary (nm)   inside <  > outside")
            if j == 0: a.set_ylabel(yl)
            if i == 0: a.set_title(t)
            a.text(0.02, 0.95, "PML body", transform=a.transAxes, fontsize=9, va="top")
    fig.suptitle("Profiles across the CLEM boundary, five bodies (IsoNet, every 3rd slice, 10 nm shells). Dotted lines: +-20 nm label uncertainty.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT + "figures/boundary_profile_5bodies.png", dpi=110)

    fig, ax = plt.subplots(2, 5, figsize=(24, 10))
    for j, t in enumerate(F.T):
        E = np.load(OUT + f"results_v2/boundary_example_{t}.npz"); pml = E["pml"]; dist = E["dist"]
        ys, xs = np.nonzero(pml); y0, y1 = max(ys.min() - 120, 0), min(ys.max() + 120, 1024); x0, x1 = max(xs.min() - 120, 100), min(xs.max() + 120, 1024)
        sl = (slice(y0, y1), slice(x0, x1))
        for i, (lay, title, cmap) in enumerate(((E["mask"], "band-passed dense mask (dark = dense)", "Blues"), (E["skel"], "fibre skeleton", "Reds"))):
            a = ax[i, j]; a.imshow(E["img"][sl], cmap="gray", vmin=-2.5, vmax=2.5)
            a.imshow(np.ma.masked_where(~lay[sl], lay[sl]), cmap=cmap, vmin=0, vmax=1.3, alpha=0.75, interpolation="nearest")
            a.contour(pml[sl], levels=[0.5], colors="yellow", linewidths=1.5); a.contour(dist[sl], levels=[-20, 20], colors="yellow", linewidths=0.7, linestyles="dotted")
            a.set_xticks([]); a.set_yticks([]); a.set_title(f"{t}: {title}", fontsize=10)
            a.plot([10, 10 + 100 / PIX], [y1 - y0 - 12] * 2, "w-", lw=3); a.text(10, y1 - y0 - 20, "100 nm", color="w", fontsize=9, fontweight="bold")
    fig.suptitle("One slice per body: CLEM label (yellow), +-20 nm (dotted). Top: dense material; bottom: fibre skeleton.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT + "figures/boundary_profile_panels.png", dpi=100)
    print("figures written")


if __name__ == "__main__":
    main()
