import numpy as np, itertools, time, zlib
def table(n): return (n >> np.arange(8)) & 1
# --- symmetry: reflection (l<->r) and complement (0<->1); Klein four-group
def reflect(n):
    t=table(n); out=0
    for i in range(8):
        l,c,r=(i>>2)&1,(i>>1)&1,i&1; j=(r<<2)|(c<<1)|l
        out|=int(t[j])<<i
    return out
def complement(n):
    t=table(n); out=0
    for i in range(8): out|=(1-int(t[7-i]))<<i
    return out
classes={}
for n in range(256):
    orbit=frozenset({n,reflect(n),complement(n),reflect(complement(n))})
    classes[min(orbit)]=sorted(orbit)
print("equivalence classes:", len(classes), "| reflect(30)=",reflect(30),"complement(30)=",complement(30),"both=",reflect(complement(30)), "| class of 110:", classes[min(110,reflect(110),complement(110),reflect(complement(110)))])
# --- Langton lambda (quiescent state 0): fraction of neighbourhoods mapping to 1 (for rules with f(000)=0)
lam=lambda n: table(n).sum()/8
# --- surjectivity via de Bruijn subset construction (Amoroso–Patt / Hedlund)
def surjective(n):
    t=table(n)
    # states = pairs (a,b) 0..3; reading output bit y, transition (a,b)->(b,c) if t[a,b,c]==y
    start=frozenset(range(4)); seen={start}; stack=[start]
    while stack:
        S=stack.pop()
        for y in (0,1):
            T=frozenset(((ab<<1)&3)|c for ab in S for c in (0,1) if t[(ab<<1)|c]==y)
            if not T: return False
            if T not in seen: seen.add(T); stack.append(T)
    return True
# --- transfer matrix preimage counting on cyclic configs
def M(n,y):
    t=table(n); m=np.zeros((4,4),dtype=np.int64)
    for a,b,c in itertools.product((0,1),repeat=3):
        if t[(a<<2)|(b<<1)|c]==y: m[(a<<1)|b,(b<<1)|c]=1
    return m
def n_preimages(n, cfg):
    P=np.eye(4,dtype=np.int64)
    for y in cfg: P=P@M(n,y)
    return int(np.trace(P))
def step(row,n):
    return table(n)[(np.roll(row,1)<<2)|(row<<1)|np.roll(row,-1)]
# brute-force verification
N=10; ok=True
for n in (30,90,110,184,0,150):
    counts={}
    for bits in itertools.product((0,1),repeat=N):
        y=tuple(step(np.array(bits),n)); counts[y]=counts.get(y,0)+1
    for cfg in [tuple(np.random.default_rng(s).integers(0,2,N)) for s in range(20)]:
        ok &= counts.get(cfg,0)==n_preimages(n,cfg)
print("transfer-matrix preimage count matches brute force (N=10):", ok)
sur=[n for n in range(256) if surjective(n)]
print("surjective rules:", len(sur), sur)
# reversible (injective) ECA: injective on all cyclic lengths -> check 1..12 and global balance
def injective_upto(n,L=12):
    for N in range(1,L+1):
        imgs=set()
        for bits in itertools.product((0,1),repeat=N):
            y=tuple(step(np.array(bits),n))
            if y in imgs: return False
            imgs.add(y)
    return True
rev=[n for n in sur if injective_upto(n,10)]
print("reversible rules:", rev)
# --- Garden of Eden count for rule 110 / 30 at N=12
for n in (30,110,90):
    goe=sum(1 for bits in itertools.product((0,1),repeat=12) if n_preimages(n,bits)==0)
    print(f"rule {n}: Garden-of-Eden configs at N=12: {goe}/4096")
# --- rule 90 closed form via Lucas' theorem
W,T=257,128; h=np.zeros((T+1,W),np.uint8); h[0,W//2]=1
for t in range(T): h[t+1]=step(h[t],90)
cf=np.zeros_like(h)
for t in range(T+1):
    for x in range(-t,t+1):
        if (t+x)%2==0:
            k=(t+x)//2
            cf[t,W//2+x]= 1 if (k & (t-k))==0 else 0
print("rule 90 Lucas closed form == simulation:", (cf==h).all())
# --- damage spreading (Lyapunov-like) for all rules
def spread(n, W=256, T=128, trials=8, seed=0):
    rng=np.random.default_rng(seed); tab=table(n); vals=[]
    for _ in range(trials):
        x=rng.integers(0,2,W).astype(np.uint8); y=x.copy(); y[W//2]^=1
        for t in range(T):
            x=tab[(np.roll(x,1)<<2)|(x<<1)|np.roll(x,-1)]; y=tab[(np.roll(y,1)<<2)|(y<<1)|np.roll(y,-1)]
        vals.append((x!=y).mean())
    return float(np.mean(vals))
def entropy_compress(n, W=256, T=256, seed=0):
    rng=np.random.default_rng(seed); tab=table(n); x=rng.integers(0,2,W).astype(np.uint8); rows=[]
    for t in range(T): x=tab[(np.roll(x,1)<<2)|(x<<1)|np.roll(x,-1)]; rows.append(x)
    arr=np.array(rows[T//2:]); b=np.packbits(arr)
    p=arr.mean(); H= 0 if p in (0,1) else -(p*np.log2(p)+(1-p)*np.log2(1-p))
    return len(zlib.compress(b.tobytes(),9))/len(b.tobytes()), H, p
t0=time.time()
rows=[]
for n in range(256):
    d=spread(n); c,H,p=entropy_compress(n); rows.append((n,lam(n),d,c,H,p))
print("atlas compute time %.1fs"%(time.time()-t0))
known={0:1,8:1,32:1,4:2,108:2,184:2,30:3,45:3,90:3,150:3,22:3,54:4,110:4}
for n,l,d,c,H,p in rows:
    if n in known: print(f"rule {n:3d} class {known[n]} lambda={l:.3f} damage={d:.3f} compress={c:.3f} density={p:.3f}")
