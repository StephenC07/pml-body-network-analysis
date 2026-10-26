#!/usr/bin/env python3
"""Round-2 controls 1 and 2, five bodies, raw and IsoNet, every 3rd slice through the body.

Control 1, fibre density with an honest denominator:
  paper skeleton (pna.segment_slice) with nothing removed and with all blobs removed;
  density = skeleton px / ANALYSED area, where analysed = compartment minus bright-mask
  pixels (the pipeline zeroes them) minus patched blob pixels.  Also a per-compartment
  threshold variant (80th percentile taken within each compartment).  Ratios inside/outside.

Control 2, dense coverage and cluster curve on a band-passed image:
  band-pass = sigma-5 minus sigma-40 Gaussian of the z-scored slice (removes brightness ramps
  and compartment offsets); noise-referenced threshold (K=1) as before; coverage in/out and
  clusters (closing + fill + label) per um^2 in size bins, >20 nm exclusion ratio.  Run with
  the standard edge mask and with a wide right mask (x >= 850 removed) for all bodies.

Output: results_v2/controls_fibre_denominator_bandpass_{src}.json, printed tables,
figures/controls_fibre_denominator_bandpass.png (after both sources are run).
"""
import json, sys, os
import numpy as np
import mrcfile
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage import exposure, morphology, restoration
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, black_tophat, disk, skeletonize
SRC = sys.argv[1] if len(sys.argv) > 1 else "isonet"
sys.argv = [sys.argv[0], SRC]
sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted")
import fibre_after_blob_removal as F
pna = F.pna; PIX = F.PIX; K, SIG = 1.0, 5.0
BINS = np.array([6, 10, 15, 20, 25, 30, 40, 60, 1000])
OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/"


def seg(sl, thr_mask=None):
    """paper pipeline; returns skeleton, bright mask; optional pixel set for the percentile."""
    D = pna.DEFAULTS
    x = pna.normalise_slice(sl.astype(np.float32)); x = restoration.denoise_tv_chambolle(x, weight=D["tv_weight"])
    bb = gaussian(x, sigma=D["bright_sigma"]); bright = bb > np.percentile(bb, D["bright_pct"])
    x = exposure.equalize_adapthist(x, clip_limit=D["clahe_clip"], kernel_size=D["clahe_kernel"])
    bth = black_tophat(x, disk(D["tophat_radius"])); bth = (bth - bth.min()) / (bth.max() - bth.min() + 1e-12)
    sel = ~bright if thr_mask is None else (~bright & thr_mask)
    thr = np.percentile(bth[sel], D["thresh_pct"]); bth[bright] = 0
    b = morphology.remove_small_objects(bth > thr, min_size=D["min_object"])
    c = morphology.remove_small_holes(binary_dilation(b, disk(D["connect_radius"])), area_threshold=D["fill_hole"])
    return skeletonize(c), bright


def noise_thr(x):
    h, e = np.histogram(x, bins=512); h = ndi.gaussian_filter1d(h.astype(float), 3)
    m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]); sd = np.sqrt(np.mean((x[x > m] - m) ** 2)); return m - K * sd


def clusters(mask, inside, outside):
    m = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2)); lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(8), np.zeros(8)
    area = np.bincount(lab.ravel())[1:]; com = np.array(ndi.center_of_mass(m, lab, range(1, n + 1))); d = 2 * np.sqrt(area / np.pi) * PIX
    cy = np.clip(np.round(com[:, 0]).astype(int), 0, 1023); cx = np.clip(np.round(com[:, 1]).astype(int), 0, 1023)
    return np.histogram(d[inside[cy, cx]], BINS)[0], np.histogram(d[outside[cy, cx]], BINS)[0]


def work(args):
    t, z, sl, art, artw, pml = args
    r = dict(t=t, z=z)
    ins, outs = pml & ~art, ~pml & ~art
    # ---- control 1
    rng = np.random.default_rng(100 * z); dmap = F.blob_diam_map(sl, art); kill = ndi.binary_dilation(dmap >= 6, iterations=2)
    for rem, img in (("none", sl), ("all", F.remove(sl, dmap, 6, rng))):
        sk, bright = seg(img); sk &= ~art
        excl = bright | (kill if rem == "all" else False)
        for w, reg in (("in", ins), ("out", outs)):
            r[f"{rem}_sk_{w}"] = int(sk[reg].sum()); r[f"{rem}_area_{w}"] = int(reg.sum()); r[f"{rem}_analysed_{w}"] = int((reg & ~excl).sum())
        ski, _ = seg(img, ins); sko, _ = seg(img, outs)
        r[f"{rem}_own_sk_in"] = int((ski & ~art)[ins].sum()); r[f"{rem}_own_sk_out"] = int((sko & ~art)[outs].sum())
    # ---- control 2
    s = (sl - sl.mean()) / (sl.std() + 1e-12); f = ndi.gaussian_filter(s, SIG); bp = f - ndi.gaussian_filter(s, 40.0)
    for name, img, a in (("global", f, art), ("bandpass", bp, art), ("bandpass_wide", bp, artw)):
        i2, o2 = pml & ~a, ~pml & ~a
        mask = (img < noise_thr(img[~a])) & ~a
        r[f"{name}_cov_in"] = float(mask[i2].mean()); r[f"{name}_cov_out"] = float(mask[o2].mean())
        ci, co = clusters(mask, i2, o2); r[f"{name}_cl_in"] = ci.tolist(); r[f"{name}_cl_out"] = co.tolist()
        r[f"{name}_area_in"] = int(i2.sum()); r[f"{name}_area_out"] = int(o2.sum())
    return r


def main():
    tasks = []
    for t, (mrc, segp, (z0, z1), mr) in F.T.items():
        pml = F.seg_load(segp); art = np.zeros((1024, 1024), bool); art[:, :100] = True
        if mr: art[:, 1024 - mr:] = True
        artw = art.copy(); artw[:, 850:] = True
        zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; zs = list(range(zidx[0], zidx[-1] + 1, 3))
        with mrcfile.open(mrc, permissive=True) as m:
            vol = m.data; zc = zs[len(zs) // 2]
            ref = mrcfile.open(F.SOURCES["isonet"][t], permissive=True).data[zc].astype(np.float32)
            sign = 1.0 if np.corrcoef(ndi.gaussian_filter(vol[zc].astype(np.float32), 3).ravel(), ndi.gaussian_filter(ref, 3).ravel())[0, 1] >= 0 else -1.0
            tasks += [(t, z, sign * vol[z].astype(np.float32), art, artw, pml[z]) for z in zs]
    with Pool(48) as pool:
        R = pool.map(work, tasks, chunksize=1)
    summary = {}
    print(f"\n##### {SRC} #####")
    print("\nCONTROL 1: fibre skeleton density ratio inside/outside")
    print(f"{'body':6s} {'nothing removed: paper':>22s} {'analysed-area':>14s} {'own-threshold':>14s} | {'all removed: paper':>18s} {'analysed-area':>14s} {'own-threshold':>14s}")
    for t in F.T:
        S = [r for r in R if r["t"] == t]; g = lambda k: sum(r[k] for r in S); row = {}
        for rem in ("none", "all"):
            row[f"{rem}_paper"] = (g(f"{rem}_sk_in") / g(f"{rem}_area_in")) / (g(f"{rem}_sk_out") / g(f"{rem}_area_out"))
            row[f"{rem}_analysed"] = (g(f"{rem}_sk_in") / g(f"{rem}_analysed_in")) / (g(f"{rem}_sk_out") / g(f"{rem}_analysed_out"))
            row[f"{rem}_own"] = (g(f"{rem}_own_sk_in") / g(f"{rem}_analysed_in")) / (g(f"{rem}_own_sk_out") / g(f"{rem}_analysed_out"))
            row[f"{rem}_analysed_frac_in"] = g(f"{rem}_analysed_in") / g(f"{rem}_area_in"); row[f"{rem}_analysed_frac_out"] = g(f"{rem}_analysed_out") / g(f"{rem}_area_out")
        summary[t] = dict(control1=row)
        print(f"{t:6s} {row['none_paper']:22.2f} {row['none_analysed']:14.2f} {row['none_own']:14.2f} | {row['all_paper']:18.2f} {row['all_analysed']:14.2f} {row['all_own']:14.2f}   (analysed fraction in/out: none {100*row['none_analysed_frac_in']:.0f}/{100*row['none_analysed_frac_out']:.0f}%, all {100*row['all_analysed_frac_in']:.0f}/{100*row['all_analysed_frac_out']:.0f}%)")
    for key, lab in (("none_paper", "nothing removed, paper"), ("none_analysed", "nothing removed, analysed area"), ("none_own", "nothing removed, own threshold"), ("all_paper", "all removed, paper"), ("all_analysed", "all removed, analysed area"), ("all_own", "all removed, own threshold")):
        v = [summary[t]["control1"][key] for t in F.T]; print(f"   five-body mean +- SD, {lab:34s}: {np.mean(v):.2f} +- {np.std(v, ddof=1):.2f}  (min {min(v):.2f})")
    print("\nCONTROL 2: dense coverage in/out and >20 nm cluster exclusion (outside/inside per um^2)")
    print(f"{'body':6s} | {'global: cov in/out':>20s} {'ratio':>6s} {'>20nm excl':>11s} | {'band-pass: cov in/out':>22s} {'ratio':>6s} {'>20nm excl':>11s} | {'band-pass wide mask: cov':>24s} {'ratio':>6s} {'>20nm excl':>11s}")
    for t in F.T:
        S = [r for r in R if r["t"] == t]; row = {}
        for name in ("global", "bandpass", "bandpass_wide"):
            ci = np.sum([r[f"{name}_cl_in"] for r in S], axis=0); co = np.sum([r[f"{name}_cl_out"] for r in S], axis=0)
            ai = sum(r[f"{name}_area_in"] for r in S) * PIX ** 2 / 1e6; ao = sum(r[f"{name}_area_out"] for r in S) * PIX ** 2 / 1e6
            cov_in = np.average([r[f"{name}_cov_in"] for r in S], weights=[r[f"{name}_area_in"] for r in S]); cov_out = np.average([r[f"{name}_cov_out"] for r in S], weights=[r[f"{name}_area_out"] for r in S])
            row[name] = dict(cov_in=float(cov_in), cov_out=float(cov_out), cov_ratio=float(cov_in / cov_out), excl20=float((co[3:].sum() / ao) / max(ci[3:].sum() / ai, 1e-9)),
                             ratio_bins=((co / ao) / np.maximum(ci / ai, 1e-9)).tolist(), n_in=ci.tolist(), n_out=co.tolist())
        summary[t]["control2"] = row
        f = lambda n: f"{100*row[n]['cov_in']:8.1f}/{100*row[n]['cov_out']:.1f}% {row[n]['cov_ratio']:6.2f} {row[n]['excl20']:11.1f}"
        print(f"{t:6s} | {f('global'):>38s} | {f('bandpass'):>40s} | {f('bandpass_wide'):>42s}")
    for name in ("global", "bandpass", "bandpass_wide"):
        cr = [summary[t]["control2"][name]["cov_ratio"] for t in F.T]; ex = [summary[t]["control2"][name]["excl20"] for t in F.T]
        print(f"   five-body mean +- SD, {name:14s}: coverage ratio {np.mean(cr):.2f} +- {np.std(cr, ddof=1):.2f}, >20 nm exclusion {np.mean(ex):.1f} +- {np.std(ex, ddof=1):.1f} (min {min(ex):.1f})")
    json.dump(summary, open(OUT + f"results_v2/controls_fibre_denominator_bandpass_{SRC}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
