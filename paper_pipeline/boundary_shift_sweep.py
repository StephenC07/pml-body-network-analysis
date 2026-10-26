#!/usr/bin/env python3
"""Boundary position sensitivity sweep (Mac, 25 Oct 2026). Same masks as boundary_profile.py (dense coverage,
>20 nm clusters) and boundary_profile_rm15.py (fibre skeleton after removing blobs >= 15 nm), inside / outside
values with the cryo-CLEM label eroded or dilated by 10 to 50 nm. Output: results_v2/boundary_shift_sweep.json.
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
OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/"
SHIFTS = [10, 20, 30, 40, 50]

def noise_thr(x):
    h, e = np.histogram(x, bins=512); h = ndi.gaussian_filter1d(h.astype(float), 3)
    m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]); sd = np.sqrt(np.mean((x[x > m] - m) ** 2)); return m - K * sd

def work(args):
    t, z, sl, art, pml = args
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
    ok = ~art; prof = {}
    regions = [("label", pml, ~pml)]
    for nm in SHIFTS:
        px = int(round(nm / PIX))
        er = ndi.binary_erosion(pml, iterations=px); di = ndi.binary_dilation(pml, iterations=px)
        regions += [(f"in{nm}", er, ~pml), (f"out{nm}", di, ~di)]
    for name, ins, outs in regions:
        ins = ins & ok; outs = outs & ok
        if ins.sum() < 500: continue
        prof[name] = dict(cov=[float(mask[ins].mean()), float(mask[outs].mean())], big=[float(big[ins].sum() / ins.sum()), float(big[outs].sum() / outs.sum())], skel=[float(skel[ins].mean()), float(skel[outs].mean())])
    return t, prof

def main():
    tasks = []
    for t, (mrc, segp, (z0, z1), mr) in F.T.items():
        pml = F.seg_load(segp); art = np.zeros((1024, 1024), bool); art[:, :100] = True
        if mr: art[:, 1024 - mr:] = True
        zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; zs = list(range(zidx[0], zidx[-1] + 1, 3))
        with mrcfile.open(mrc, permissive=True) as m:
            tasks += [(t, z, m.data[z].astype(np.float32), art, pml[z]) for z in zs]
    with Pool(8) as pool:
        R = pool.map(work, tasks, chunksize=1)
    names = ["label"] + [f"{d}{nm}" for nm in SHIFTS for d in ("in", "out")]
    results = {}
    for t in F.T:
        P = [p for tt, p in R if tt == t]; results[t] = {}
        for name in names:
            Q = [p[name] for p in P if name in p]
            if not Q: continue
            results[t][name] = {k: [float(np.mean([q[k][0] for q in Q])), float(np.mean([q[k][1] for q in Q]))] for k in ("cov", "big", "skel")}
            results[t][name]["n_slices"] = len(Q)
        print(t, {name: round(results[t][name]["skel"][0] / results[t][name]["skel"][1], 2) for name in names if name in results[t]})
    json.dump(results, open(OUT + "results_v2/boundary_shift_sweep.json", "w"), indent=1)

if __name__ == "__main__":
    main()
