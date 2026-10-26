"""Compare the interior fibre network density between the five PML bodies (per slice, step 3: blobs >= 15 nm removed),
and the same for the surrounding nucleoplasm. Writes results_v2/body_vs_body_fibre.json and figures/body_vs_body_fibre.png/pdf."""
import json, sys
import numpy as np, mrcfile
from multiprocessing import Pool
from scipy import ndimage as ndi, stats
sys.argv = [sys.argv[0], "isonet"]; sys.path.insert(0, "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted")
import fibre_after_blob_removal as F
pna = F.pna; PIX = F.PIX; OUT = "/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/"
def work(a):
    t, z, sl, art, pml_z = a
    rng = np.random.default_rng(100 * z); dmap = F.blob_diam_map(sl, art)
    for dmin in (30, 20, 15): img = F.remove(sl, dmap, dmin, rng)     # same rng consumption order as the main script
    skel, _ = pna.segment_slice(img); skel &= ~art
    ins = pml_z & ~art; outs = ~pml_z & ~art
    return t, z, float(skel[ins].mean()), float(skel[outs].mean()), int(ins.sum())
def main():
    tasks = []
    for t, (mrc, seg, (z0, z1), mr) in F.T.items():
        pml = F.seg_load(seg); art = np.zeros((1024, 1024), bool); art[:, :100] = True
        if mr: art[:, 1024 - mr:] = True
        zidx = np.where(pml[z0:z1 + 1].sum(axis=(1, 2)) > 0)[0] + z0; zs = list(range(zidx[0], zidx[-1] + 1, 3))
        with mrcfile.open(mrc, permissive=True) as m: tasks += [(t, z, m.data[z].astype(np.float32), art, pml[z]) for z in zs]
    with Pool(8) as pool: R = pool.map(work, tasks, chunksize=1)
    res = {t: dict(z=[], fin=[], fout=[], area=[]) for t in F.T}
    for t, z, fi, fo, a in R: res[t]["z"].append(z); res[t]["fin"].append(100 * fi); res[t]["fout"].append(100 * fo); res[t]["area"].append(a)
    json.dump(res, open(OUT + "results_v2/body_vs_body_fibre.json", "w"), indent=1)
    B = list(F.T)
    for key, lab in (("fin", "INSIDE"), ("fout", "OUTSIDE")):
        groups = [np.array(res[t][key]) for t in B]
        print(f"\n{lab} fibre density per slice (% area): " + "  ".join(f"{t} {np.median(g):.2f} [{np.percentile(g,25):.2f}-{np.percentile(g,75):.2f}] n={len(g)}" for t, g in zip(B, groups)))
        print(f"  Kruskal-Wallis H={stats.kruskal(*groups).statistic:.1f} p={stats.kruskal(*groups).pvalue:.2e}")
        pv = []
        for i in range(5):
            for j in range(i + 1, 5): pv.append((B[i], B[j], stats.mannwhitneyu(groups[i], groups[j]).pvalue))
        m = len(pv); order = sorted(range(m), key=lambda k: pv[k][2]); holm = [None] * m
        for rank, k in enumerate(order): holm[k] = min(1.0, pv[k][2] * (m - rank))
        print("  pairwise Mann-Whitney (Holm): " + ", ".join(f"{a} vs {b} p={h:.3f}" for (a, b, _), h in zip(pv, holm)))
if __name__ == "__main__": main()
