import numpy as np, time, re
def parse(s):
    m=re.fullmatch(r"B([0-8]*)/S([0-8]*)",s); return [int(c) for c in m[1]],[int(c) for c in m[2]]
def lut(B,S):
    t=np.zeros((2,9),np.uint8); t[0,B]=1; t[1,S]=1; return t
def counts(g): 
    ax=(-2,-1); return sum(np.roll(np.roll(g,dy,ax[0]),dx,ax[1]) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx)
def step(g,t): return t[g,counts(g)]
def infer(x0,x1,eps=0.0):
    c=counts(x0); k1=np.zeros((2,9)); k0=np.zeros((2,9))
    np.add.at(k1,(x0,c),x1); np.add.at(k0,(x0,c),1-x1)
    if eps==0: llr=np.where(k1>0,np.inf,0)-np.where(k0>0,np.inf,0)
    else: llr=(k1-k0)*np.log((1-eps)/eps)
    p=1/(1+np.exp(-np.clip(np.nan_to_num(llr,posinf=50,neginf=-50),-50,50)))
    p[(k1+k0)==0]=0.5
    return p, k1+k0
rng=np.random.default_rng(0)
for s in ["B3/S23","B36/S23","B3678/S34678","B2/S","B35678/S5678"]:
    T=lut(*parse(s)); x0=(rng.random((64,64))<0.5).astype(np.uint8); x1=step(x0,T)
    p,n=infer(x0,x1); est=(p>0.5).astype(int); unk=(n==0)
    noisy=x1^(rng.random(x1.shape)<0.05); pn,_=infer(x0,noisy,0.05)
    print(f"{s:14s} exact recovered={((est==T)|unk).all()} unobserved_entries={int(unk.sum())} ({np.argwhere(unk).tolist()}) | 5% noise bit errors={int(((pn>0.5)!=T)[~unk].sum())}")
# exhaustive search over all 2^18 rules for k=2 on 24x24
R=1<<18; bits=((np.arange(R)[:,None]>>np.arange(18))&1).astype(np.uint8).reshape(R,2,9)
Ttrue=lut(*parse("B3/S23")); x0=(rng.random((24,24))<0.4).astype(np.uint8); x2=step(step(x0,Ttrue),Ttrue)
t0=time.time(); hits=[]
for lo in range(0,R,8192):
    L=bits[lo:lo+8192]; ridx=np.arange(len(L))[:,None,None]
    g=np.broadcast_to(x0,(len(L),24,24)).astype(np.uint8)
    g=L[ridx,g,counts(g)]; g=L[ridx,g,counts(g)]
    ok=(g==x2).all((1,2)); hits+= (np.nonzero(ok)[0]+lo).tolist()
print(f"exhaustive k=2 search over 2^18 rules: {time.time()-t0:.1f}s, consistent rules={len(hits)}")
cons=bits[hits]; print("bits fixed across all consistent rules:", int((cons.min(0)==cons.max(0)).sum()),"/18")
# duality check: Day&Night self-dual; Life dual
def dual(B,S): return sorted(8-n for n in range(9) if n not in S), sorted(8-n for n in range(9) if n not in B)
B,S=parse("B3/S23"); Bd,Sd=dual(B,S); print("dual of Life: B%s/S%s"%("".join(map(str,Bd)),"".join(map(str,Sd))), "| Day&Night self-dual:", dual(*parse("B3678/S34678"))==tuple(map(sorted,parse("B3678/S34678"))))
g=(rng.random((40,40))<0.5).astype(np.uint8)
print("duality verified by simulation:", (1-step(g,lut(B,S)) == step(1-g,lut(Bd,Sd))).all())
