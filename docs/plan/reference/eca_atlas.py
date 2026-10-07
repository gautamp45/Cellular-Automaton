import numpy as np, zlib, time
def table(n): return (n >> np.arange(8)) & 1
def ev(x,tab): return tab[(np.roll(x,1)<<2)|(x<<1)|np.roll(x,-1)]
def metrics(n, W=251, T=256, trials=6, seed=0):
    rng=np.random.default_rng(seed); tab=table(n); dmg=[]; comp=[]; dens=[]
    for _ in range(trials):
        x=rng.integers(0,2,W).astype(np.uint8); y=x.copy(); y[W//2]^=1; rows=[]; d=[]
        for t in range(T):
            x=ev(x,tab); y=ev(y,tab)
            if t>=T//2: rows.append(x); d.append((x!=y).mean())
        arr=np.array(rows); b=np.packbits(arr).tobytes()
        dmg.append(np.mean(d)); comp.append(len(zlib.compress(b,9))/len(b)); dens.append(arr.mean())
    return float(np.mean(dmg)), float(np.mean(comp)), float(np.mean(dens))
t0=time.time(); R=np.array([metrics(n) for n in range(256)]); print("time %.1fs"%(time.time()-t0))
known={0:1,8:1,32:1,40:1,4:2,108:2,184:2,1:2,30:3,45:3,90:3,150:3,22:3,18:3,126:3,54:4,110:4,106:4}
for n in sorted(known): print(f"rule {n:3d} class {known[n]} damage={R[n,0]:.3f} compress={R[n,1]:.3f} density={R[n,2]:.3f}")
# simple unsupervised: kmeans k=4 on (damage, compress)
X=R[:,:2]; rng=np.random.default_rng(0)
best=None
for rs in range(20):
    C=X[rng.choice(256,4,replace=False)]
    for _ in range(100):
        lab=((X[:,None]-C[None])**2).sum(-1).argmin(1)
        C=np.array([X[lab==k].mean(0) if (lab==k).any() else C[k] for k in range(4)])
    inertia=((X-C[lab])**2).sum()
    if best is None or inertia<best[0]: best=(inertia,lab.copy(),C.copy())
_,lab,C=best; order=np.argsort(C[:,1]); remap={o:i for i,o in enumerate(order)}
print("cluster centroids (damage, compress) sorted by compress:", np.round(C[order],3).tolist())
for n in sorted(known): print(n, "known", known[n], "cluster", remap[lab[n]])
print("cluster sizes:", np.bincount([remap[l] for l in lab]).tolist())
