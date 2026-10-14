import numpy as np, re
def table(n): return (n >> np.arange(8)) & 1
def estep(row, n, wrap=True):
    l = np.roll(row,1); r = np.roll(row,-1)
    if not wrap: l[0]=0; r[-1]=0
    return table(n)[(l<<2)|(row<<1)|r].astype(np.uint8)
def erun(n, w, g, wrap=True):
    out=np.zeros((g+1,w),np.uint8); out[0,w//2]=1
    for t in range(g): out[t+1]=estep(out[t],n,wrap)
    return out
print("table30", table(30).tolist(), "table110", table(110).tolist(), "table90", table(90).tolist())
r30=erun(30,201,40); print("rule30 centre col 0..24:", r30[:25,100].tolist())
r90=erun(90,129,32); print("rule90 popcount ok:", all(r90[t].sum()==2**bin(t).count('1') for t in range(32)))
print("rule30 row1 around centre:", r30[1,97:104].tolist(), "row2:", r30[2,96:105].tolist())
print("rule110 single cell 1 step:", erun(110,11,1)[1].tolist())

def parse_rule(s):
    m=re.fullmatch(r"[Bb]([0-8]*)/?[Ss]([0-8]*)", s.strip()); return set(map(int,m[1])), set(map(int,m[2]))
def lut(B,S):
    t=np.zeros((2,9),np.uint8); t[0,list(B)]=1; t[1,list(S)]=1; return t
def counts(g, wrap=True):
    if wrap: return sum(np.roll(np.roll(g,dy,0),dx,1) for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0))
    p=np.pad(g,1); H,W=g.shape; return sum(p[1+dy:1+dy+H,1+dx:1+dx+W] for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0))
def lstep(g,B,S,wrap=True): return lut(B,S)[g, counts(g.astype(np.uint8),wrap)]
def rle(s):
    rows=[[]]; n=''
    for ch in s.strip().rstrip('!'):
        if ch.isdigit(): n+=ch; continue
        k=int(n or 1); n=''
        if ch=='b': rows[-1]+= [0]*k
        elif ch=='o': rows[-1]+= [1]*k
        elif ch=='$': rows += [[] for _ in range(k)]
    w=max(map(len,rows)); return np.array([r+[0]*(w-len(r)) for r in rows],np.uint8)
P={"glider":"bo$2bo$3o!","blinker":"3o!","toad":"b3o$3o!","beacon":"2o$2o$2b2o$2b2o!","lwss":"bo2bo$o4b$o3bo$4o!",
   "r_pentomino":"b2o$2o$bo!","acorn":"bo5b$3bo3b$2o2b3o!","diehard":"6bob$2o6b$bo3b3o!",
   "pulsar":"2b3o3b3o2b2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2b2$2b3o3b3o2b$o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!",
   "gosper_glider_gun":"24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!"}
LIFE=({3},{2,3})
def place(p, H=64, W=64):
    g=np.zeros((H,W),np.uint8); y=(H-p.shape[0])//2; x=(W-p.shape[1])//2; g[y:y+p.shape[0],x:x+p.shape[1]]=p; return g
for name,s in P.items():
    p=rle(s); g0=place(p, 80, 80) if name!="diehard" else place(p,120,120); g=g0.copy(); pops=[int(g.sum())]
    hist=[g0]
    for t in range(1,131):
        g=lstep(g,*LIFE,wrap=False); pops.append(int(g.sum())); hist.append(g)
    per=next((k for k in range(1,16) if (hist[k]==g0).all()),None)
    shift=None
    if name in("glider","lwss"):
        k=4
        ys,xs=np.nonzero(g0); ys2,xs2=np.nonzero(hist[k])
        shift=(int(ys2.min()-ys.min()),int(xs2.min()-xs.min())) if (np.roll(np.roll(g0,ys2.min()-ys.min(),0),xs2.min()-xs.min(),1)==hist[k]).all() else "nomatch"
    print(f"{name:18s} shape={p.shape} pop0={pops[0]} period={per} shift@4={shift} pop@30={pops[30]} pop@130={pops[130]}")
# diehard dies at 130
g=place(rle(P["diehard"]),120,120)
for t in range(1,131): g=lstep(g,*LIFE,wrap=False); 
print("diehard alive at 129?", ); g=place(rle(P["diehard"]),120,120)
for t in range(129): g=lstep(g,*LIFE,wrap=False)
print(" pop@129", int(g.sum()), "pop@130", int(lstep(g,*LIFE,wrap=False).sum()))
# gun: population after 30 gens on large board: period-30 gun emits glider
g=place(rle(P["gosper_glider_gun"]),80,80); g0=g.copy()
for t in range(30): g=lstep(g,*LIFE,wrap=False)
print("gun pop0",int(g0.sum()),"pop30",int(g.sum()),"(should be pop0+5)")
# seeds B2/S: blinker dies; edge case wrap vs dead
b=place(rle("3o!"),5,5); print("blinker wrap 5x5 ok:", (lstep(lstep(b,*LIFE),*LIFE)==b).all())
