#!/usr/bin/env python3
"""Fibre (fine-texture) analysis after progressive removal of dense blobs, five bodies.

Blob detection: as blob_size_exclusion.py (IsoNet tomograms, sigma 5 px, noise-referenced
K = 1.0, closed + filled, size = equivalent diameter). Blobs are REMOVED by replacing their
pixels with the local background (sigma-10 blurred image) plus matched noise, so the
skeleton pipeline cannot trace them.

Steps (cumulative):
  0  nothing removed
  1  blobs >= 30 nm removed
  2  blobs >= 20 nm removed
  3  blobs >= 15 nm removed
  4  blobs >=  6 nm removed  (all detected blobs)

Fibre measure: the paper's pipeline (pml_network_analysis.segment_slice) on every 3rd slice
through the body Z range -> skeleton density and branch density per 12x12 tile -> the
paper's composite score; reported as mean score in PML tiles vs non-PML tiles, Pearson r
against PML fraction, and raw skeleton density inside vs outside (px per px).
"""
import json, sys
import numpy as np
import mrcfile, tifffile
from multiprocessing import Pool
from scipy import ndimage as ndi
sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-body-network-analysis")
import pml_network_analysis as pna

PIX = 0.992; K, SIG = 1.0, 5.0
SRC = sys.argv[1] if len(sys.argv) > 1 else "isonet"
SOURCES = {"isonet": {"TS_01": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/isonet_pml_deconv_2/2_binnedtc.mrc", "TS_03": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/isonet_pml_deconv_2/3_1_binnedtc.mrc",
                      "TS_08": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/isonet_pml_deconv_2/8_binnedtc.mrc", "TS_15": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/isonet_pml_deconv_2/darkthreshold0815_.mrc",
                      "TS_16": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/isonet_pml_deconv_2/darkthreshold0816.mrc"},
           "raw": {"TS_01": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/raw_aretomo2/2_binnedtc.mrc", "TS_03": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/raw_aretomo2/3_1_binnedtc.mrc",
                   "TS_08": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/raw_aretomo2/8_binnedtc.mrc", "TS_15": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/raw_aretomo2/darkthreshold0815.mrc",
                   "TS_16": "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/raw_aretomo2/darkthreshold0816.mrc"},
           "double": {t: f"/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/isonet2_run/isonet_maps/corrected_epochs50_isonet2_unet-medium_{t}.mrc" for t in ("TS_01", "TS_03", "TS_08")}}
D = None
T_ALL = {"TS_01": ("2_binnedtc.mrc", "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/segmentations/2/2_binnedtc(3).labels.tif", (131, 261), 60),
     "TS_03": ("3_1_binnedtc.mrc", "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/segmentations/3/PML_area.labels.tiff", (82, 245), 60),
     "TS_08": ("8_binnedtc.mrc", "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/segmentations/8/8_binnedtc.labels.tif", (85, 249), 60),
     "TS_15": ("darkthreshold0815_.mrc", "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/segmentations/15/darkthreshold0815.labels.tif", (91, 261), 60),
     "TS_16": ("darkthreshold0816.mrc", "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/segmentations/16/darkthreshold0816.obj.voxelised_bin4.npy", (64, 242), 60)}
T = {t: (SOURCES[SRC][t], v[1], v[2], v[3]) for t, v in T_ALL.items() if t in SOURCES[SRC]}
STEPS = [("0 nothing removed", 1e9), ("1 >=30 nm removed", 30), ("2 >=20 nm removed", 20), ("3 >=15 nm removed", 15), ("4 all blobs removed", 6)]
K8 = np.ones((3, 3), int)


def seg_load(p):
    return np.load(p) if p.endswith(".npy") else (tifffile.imread(p) == 2)


def blob_diam_map(sl, art):
    s = (sl - sl.mean()) / (sl.std() + 1e-12); f = ndi.gaussian_filter(s, SIG)
    x = f[~art]; h, e = np.histogram(x, bins=512); h = ndi.gaussian_filter1d(h.astype(float), 3)
    m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]); sd = np.sqrt(np.mean((x[x > m] - m) ** 2))
    mask = (f < m - K * sd) & ~art; mask = ndi.binary_closing(mask, iterations=2); mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask); area = np.bincount(lab.ravel()); d = 2 * np.sqrt(area / np.pi) * PIX; d[0] = 0
    return d[lab]


def remove(sl, dmap, dmin, rng):
    """Replace pixels of blobs with diameter >= dmin by local background + matched noise."""
    kill = ndi.binary_dilation(dmap >= dmin, iterations=2)
    if not kill.any():
        return sl
    bg = ndi.gaussian_filter(sl, 10.0)
    resid = sl - bg; noise_sd = resid[~kill].std()
    out = sl.copy(); out[kill] = bg[kill] + rng.normal(0, noise_sd, kill.sum()).astype(np.float32)
    return ndi.gaussian_filter(out, 1.0) * kill + sl * (~kill)   # light smoothing of the patch only


def conn_metrics(skel, region):
    """Connectivity of skeleton pieces whose centroid lies in region."""
    from skimage.measure import label, regionprops_table
    lab = label(skel, connectivity=2)
    if lab.max() == 0:
        return dict(n=0, L=0.0, L30=0.0, L50=0.0, big=0.0, junc=0)
    P = regionprops_table(lab, properties=("area", "centroid", "major_axis_length"))
    cy = np.clip(np.round(P["centroid-0"]).astype(int), 0, 1023); cx = np.clip(np.round(P["centroid-1"]).astype(int), 0, 1023)
    sel = region[cy, cx]; a = P["area"][sel].astype(float); ext = P["major_axis_length"][sel] * PIX
    nb = ndi.convolve(skel.astype(int), K8, mode="constant") * skel
    junc = int(((nb >= 3) & region).sum())
    tot = a.sum()
    return dict(n=int(sel.sum()), L=float(tot), L30=float(a[ext >= 30].sum()), L50=float(a[ext >= 50].sum()),
                big=float(a.max() if len(a) else 0), junc=junc)


def work(args):
    t, z, sl, art, seed, pml_z = args
    rng = np.random.default_rng(seed); dmap = blob_diam_map(sl, art); out = {}
    ins = pml_z & ~art; outs = ~pml_z & ~art
    for name, dmin in STEPS:
        img = remove(sl, dmap, dmin, rng)
        skel, br = pna.segment_slice(img)
        skel &= ~art; br &= ~art
        out[name] = (skel, br, conn_metrics(skel, ins), conn_metrics(skel, outs))
    return t, z, out


def main():
    results = {}
    for t, (mrc, seg, (z0, z1), mr) in T.items():
        pml = seg_load(seg); art = np.zeros((1024, 1024), bool); art[:, :100] = True
        if mr: art[:, 1024 - mr:] = True
        zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; zs = list(range(zidx[0], zidx[-1] + 1, 3))
        with mrcfile.open(mrc, permissive=True) as m:
            vol = m.data
            zc = zs[len(zs) // 2]
            ref = mrcfile.open(SOURCES["isonet"][t], permissive=True).data[zc].astype(np.float32)
            sign = 1.0 if np.corrcoef(ndi.gaussian_filter(vol[zc].astype(np.float32), 3).ravel(), ndi.gaussian_filter(ref, 3).ravel())[0, 1] >= 0 else -1.0
            tasks = [(t, z, sign * vol[z].astype(np.float32), art, 100 * z, pml[z]) for z in zs]
        with Pool(40) as pool:
            res = pool.map(work, tasks, chunksize=1)
        gh, gw = 1024 // 12, 1024 // 12
        pml_grid = np.array([[pml[zs][:, i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean() for j in range(12)] for i in range(12)])
        ins = pml[zs] & ~art; outs = ~pml[zs] & ~art
        results[t] = {}
        print(f"\n=== {t} ({SRC}, {len(zs)} slices) ===")
        print(f"{'step':24s} {'skel in':>8s} {'skel out':>9s} {'ratio':>6s} | {'score PML':>9s} {'non-PML':>8s} {'r':>6s} | connectivity in/out: mean piece length, share of skeleton in pieces >=30 nm and >=50 nm, junctions per um of skeleton")
        for name, _ in STEPS:
            skel = np.array([r[2][name][0] for r in res]); br = np.array([r[2][name][1] for r in res])
            fib = np.array([[skel[:, i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean() for j in range(12)] for i in range(12)])
            bra = np.array([[br[:, i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean() for j in range(12)] for i in range(12)])
            fn = (fib - fib.min()) / (fib.max() - fib.min() + 1e-12); bn = (bra - bra.min()) / (bra.max() - bra.min() + 1e-12)
            comp = 0.6 * fn + 0.4 * bn
            r = float(np.corrcoef(comp.ravel(), pml_grid.ravel())[0, 1])
            si, so = float(skel[ins].mean()), float(skel[outs].mean())
            results[t][name] = dict(skel_in=si, skel_out=so, ratio=si / so, score_pml=float(comp[pml_grid > 0].mean()), score_non=float(comp[pml_grid == 0].mean()), r=r)
            ci = [r_[2][name][2] for r_ in res]; co = [r_[2][name][3] for r_ in res]
            def agg(c, key): return sum(x[key] for x in c)
            Li, Lo = agg(ci, "L"), agg(co, "L")
            mi = dict(seg_len_nm=Li * PIX / max(agg(ci, "n"), 1), frac30=agg(ci, "L30") / max(Li, 1), frac50=agg(ci, "L50") / max(Li, 1),
                      junc_per_um=agg(ci, "junc") / (Li * PIX / 1000), pieces_per_um2=agg(ci, "n") / (ins.sum() * PIX ** 2 / 1e6))
            mo = dict(seg_len_nm=Lo * PIX / max(agg(co, "n"), 1), frac30=agg(co, "L30") / max(Lo, 1), frac50=agg(co, "L50") / max(Lo, 1),
                      junc_per_um=agg(co, "junc") / (Lo * PIX / 1000), pieces_per_um2=agg(co, "n") / (outs.sum() * PIX ** 2 / 1e6))
            results[t][name].update(conn_in=mi, conn_out=mo)
            print(f"{name:24s} {si:8.4f} {so:9.4f} {si/so:6.2f} | {comp[pml_grid>0].mean():9.2f} {comp[pml_grid==0].mean():8.2f} {r:6.2f} | "
                  f"piece len {mi['seg_len_nm']:5.1f}/{mo['seg_len_nm']:5.1f} nm  >=30nm {100*mi['frac30']:4.1f}/{100*mo['frac30']:4.1f}%  >=50nm {100*mi['frac50']:4.1f}/{100*mo['frac50']:4.1f}%  junctions/um {mi['junc_per_um']:5.1f}/{mo['junc_per_um']:5.1f}", flush=True)
    json.dump(results, open(f"/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/results_v2/fibre_after_blob_removal_{SRC}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
