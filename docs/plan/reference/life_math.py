import numpy as np, time, zlib
from math import comb
def parse(s):
    import re; m=re.fullmatch(r"B([0-8]*)/S([0-8]*)",s); return [int(c) for c in m[1]],[int(c) for c in m[2]]
def mf_map(rho,B,S):
    p=[comb(8,n)*rho**n*(1-rho)**(8-n) for n in range(9)]
    return rho*sum(p[n] for n in S)+(1-rho)*sum(p[n] for n in B)
def mf_fixed(B,S,rho0=0.5,it=2000):
    r=rho0
    for _ in range(it): r=mf_map(r,B,S)
    return r
def batch_sim(luts, H=64, W=64, T=200, density=0.5, seed=0):
    # luts: (R,2,9) uint8 ; grids (R,H,W)
    R=len(luts); rng=np.random.default_rng(seed); g=(rng.random((R,H,W))<density).astype(np.uint8)
    ridx=np.arange(R)[:,None,None]; dens=[]; act=[]
    for t in range(T):
        c=sum(np.roll(np.roll(g,dy,1),dx,2) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx)
        ng=luts[ridx,g,c]
        act.append((ng!=g).mean((1,2))); g=ng; dens.append(g.mean((1,2)))
    return np.array(dens).T, np.array(act).T, g
def lut(B,S):
    t=np.zeros((2,9),np.uint8); t[0,B]=1; t[1,S]=1; return t
cat={"life":"B3/S23","highlife":"B36/S23","day_night":"B3678/S34678","seeds":"B2/S","maze":"B3/S12345","coral":"B3/S45678","diamoeba":"B35678/S5678","replicator":"B1357/S1357","anneal":"B4678/S35678","34life":"B34/S34","long_life":"B345/S5","morley":"B368/S245"}
luts=np.array([lut(*parse(s)) for s in cat.values()])
d,a,_=batch_sim(luts,128,128,400,0.5,0)
for i,(k,s) in enumerate(cat.items()):
    B,S=parse(s); print(f"{k:11s} {s:14s} mean-field rho*={mf_fixed(B,S):.3f}  simulated rho(t=400)={d[i,-1]:.3f}  activity={a[i,-50:].mean():.3f}")
# timing for rule-space sweep
rng=np.random.default_rng(1); R=512
rand_luts=rng.integers(0,2,(R,2,9)).astype(np.uint8); rand_luts[:,0,0]=0  # exclude B0 (strobing)
t0=time.time(); d,a,g=batch_sim(rand_luts,64,64,200,0.5,0); dt=time.time()-t0
print(f"batch {R} rules 64x64x200 steps: {dt:.1f}s  -> 4096 rules ~{dt*8/60:.1f} min; all 2^17 non-B0 rules ~{dt*256/3600:.1f} h")
