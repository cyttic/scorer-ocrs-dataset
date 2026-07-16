#!/usr/bin/env python3
"""Dataset scorecard: compare synthetic OCR datasets against a real target (Matan) in a
frozen encoder's embedding space. See IDEAS.md for the rationale.

    /mnt/ssd2/cyttic/ml_env/bin/python compare_datasets.py \
        --ref matan=/path/to/matan_test.parquet \
        --data composed250k=/mnt/ssd2/cyttic/datasets/heb_bigram_composed_250k \
        --data connected3=/mnt/ssd2/cyttic/datasets/heb_connected_composed_3track \
        --n 2000 --encoder mobilenet

Outputs a table: KID->ref, Coverage, Density, Vendi, mean pairwise dist, proxy A-distance.
Accepts a parquet file or a directory of parquet shards per dataset (image column = HF-style
dict with 'bytes', or raw bytes).
"""
import argparse, glob, io, os, random, sys
import numpy as np

def load_images(spec, n, seed=42):
    """spec = parquet file or dir; sample n PIL images."""
    import pandas as pd
    from PIL import Image
    files = sorted(glob.glob(os.path.join(spec, "*.parquet"))) if os.path.isdir(spec) else [spec]
    assert files, f"no parquet under {spec}"
    rng = random.Random(seed)
    rng.shuffle(files)
    per = max(1, n // min(len(files), 8)) + 1
    out = []
    for f in files[:8]:
        df = pd.read_parquet(f, columns=["image"])
        idx = list(range(len(df))); rng.shuffle(idx)
        for i in idx[:per]:
            cell = df["image"].iloc[i]
            b = cell["bytes"] if isinstance(cell, dict) else cell
            try: out.append(Image.open(io.BytesIO(b)).convert("L"))
            except Exception: pass
            if len(out) >= n: break
        if len(out) >= n: break
    return out[:n]

def patches(img, side, max_p=3, min_ink=0.01):
    """resize to height=side, slice up to max_p square windows, drop near-empty."""
    from PIL import Image
    w, h = img.size
    nw = max(side, int(w * side / h))
    img = img.resize((nw, side), Image.LANCZOS)
    xs = [0] if nw <= side else list(np.linspace(0, nw - side, min(max_p, 1 + nw // side)).astype(int))
    res = []
    for x in xs:
        p = img.crop((x, 0, x + side, side))
        if (np.asarray(p) < 200).mean() >= min_ink:
            res.append(p.convert("RGB"))
    return res

@np.errstate(all="ignore")
def _noop(): pass

def embed_all(imgs, encoder, device, bs=64):
    import torch, timm
    from timm.data import resolve_data_config, create_transform
    kw = {"img_size": 224} if "dinov2" in encoder else {}
    model = timm.create_model(encoder, pretrained=True, num_classes=0, **kw).to(device).eval()
    cfg = resolve_data_config({}, model=model)
    side = cfg["input_size"][1]
    tf = create_transform(**cfg)
    feats = []
    batch = []
    with torch.no_grad():
        def flush():
            if not batch: return
            x = torch.stack(batch).to(device)
            f = model(x).float().cpu().numpy()
            feats.append(f); batch.clear()
        for im in imgs:
            for p in patches(im, side):
                batch.append(tf(p))
                if len(batch) >= bs: flush()
        flush()
    E = np.concatenate(feats) if feats else np.zeros((0, 1))
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E

# ---------------- metrics ----------------
def kid(X, Y, subset=1000, reps=10, seed=0):
    """unbiased MMD^2 with polynomial kernel (Binkowski et al.), x1000"""
    d = X.shape[1]; rng = np.random.default_rng(seed); vals = []
    def k(a, b): return (a @ b.T / d + 1.0) ** 3
    for _ in range(reps):
        x = X[rng.choice(len(X), min(subset, len(X)), replace=False)]
        y = Y[rng.choice(len(Y), min(subset, len(Y)), replace=False)]
        m, n = len(x), len(y)
        kxx = k(x, x); kyy = k(y, y); kxy = k(x, y)
        np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum()/(m*(m-1)) + kyy.sum()/(n*(n-1)) - 2*kxy.mean())
    return float(np.mean(vals))*1000, float(np.std(vals))*1000

def coverage_density(X, R, k=5):
    """Naeem et al.: R=real(ref) manifold, X=candidate."""
    from sklearn.neighbors import NearestNeighbors
    nnR = NearestNeighbors(n_neighbors=k+1).fit(R)
    radii = nnR.kneighbors(R)[0][:, -1]                    # kth-NN radius per real point
    d_xr = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(R)[0][:, 0]
    cov = float((d_xr < radii).mean())                     # real points with a synth neighbor inside their radius
    nnX = NearestNeighbors(n_neighbors=1).fit(R)
    dist, idx = nnX.kneighbors(X)
    dens = float((dist[:, 0] < radii[idx[:, 0]]).mean())   # synth points landing inside a real ball
    return cov, dens

def vendi(X, cap=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = X[rng.choice(len(X), min(cap, len(X)), replace=False)]
    K = x @ x.T / len(x)
    ev = np.linalg.eigvalsh(K); ev = np.clip(ev, 1e-12, None); ev /= ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))

def mean_pairwise(X, cap=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = X[rng.choice(len(X), min(cap, len(X)), replace=False)]
    S = x @ x.T
    n = len(x); off = (S.sum() - np.trace(S)) / (n*(n-1))
    return float(1 - off)                                   # mean cosine distance

def a_distance(X, R, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    n = min(len(X), len(R))
    rng = np.random.default_rng(seed)
    Z = np.concatenate([X[rng.choice(len(X), n, replace=False)],
                        R[rng.choice(len(R), n, replace=False)]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    acc = cross_val_score(LogisticRegression(max_iter=2000), Z, y, cv=5).mean()
    return float(2*(2*acc - 1)), float(acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="name=path of the REAL target (e.g. matan=..parquet)")
    ap.add_argument("--data", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--n", type=int, default=2000, help="images sampled per dataset")
    ap.add_argument("--encoder", default="mobilenet",
                    choices=["mobilenet", "dinov2"],)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="tsv output path")
    a = ap.parse_args()
    enc = {"mobilenet": "mobilenetv2_100.ra_in1k",
           "dinov2": "vit_small_patch14_dinov2.lvd142m"}[a.encoder]

    rname, rpath = a.ref.split("=", 1)
    print(f"encoder={enc} | n={a.n}/dataset | ref={rname}:{rpath}", flush=True)
    print(f"[embed] {rname}...", flush=True)
    R = embed_all(load_images(rpath, a.n), enc, a.device)
    print(f"  ref patches: {len(R)}", flush=True)

    rows = []
    for spec in a.data:
        name, path = spec.split("=", 1)
        print(f"[embed] {name}...", flush=True)
        X = embed_all(load_images(path, a.n), enc, a.device)
        print(f"  patches: {len(X)}", flush=True)
        kid_m, kid_s = kid(X, R)
        cov, dens = coverage_density(X, R)
        ad, acc = a_distance(X, R)
        rows.append((name, kid_m, kid_s, cov, dens, vendi(X), mean_pairwise(X), ad, acc, len(X)))
    # gold row: ref vs itself split in half
    h = len(R)//2
    kid_m, kid_s = kid(R[:h], R[h:])
    cov, dens = coverage_density(R[:h], R[h:])
    ad, acc = a_distance(R[:h], R[h:])
    rows.append((f"{rname}(self-split)", kid_m, kid_s, cov, dens, vendi(R), mean_pairwise(R), ad, acc, len(R)))

    hdr = f"{'dataset':24} {'KID->ref':>10} {'±':>6} {'Cover':>6} {'Dens':>6} {'Vendi':>7} {'PairD':>6} {'A-dist':>7} {'clfAcc':>7} {'#patch':>7}"
    print("\n" + hdr); print("-"*len(hdr))
    lines = [hdr]
    for r in rows:
        ln = f"{r[0]:24} {r[1]:10.2f} {r[2]:6.2f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:7.1f} {r[6]:6.3f} {r[7]:7.3f} {r[8]:7.3f} {r[9]:7d}"
        print(ln); lines.append(ln)
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")
        print("\nsaved ->", a.out)

if __name__ == "__main__":
    main()
