import sys, time, numpy as np, glob
sys.path.insert(0,__import__('os').path.dirname(__file__)); from np_nca import load_pt, FILTERS
from PIL import Image
def kernels(f):
    k,s=FILTERS[f]; k=np.array(k,np.float32)/s; i=np.zeros((3,3),np.float32); i[1,1]=1; return np.stack([i,k,k.T])
def make_step(sd,filt):
    K=kernels(filt); w1=sd["update_module.0.weight"][:,:,0,0]; b1=sd["update_module.0.bias"][:,None]; w2=sd["update_module.2.weight"][:,:,0,0]
    def alive(a):
        p=np.pad(a,1,constant_values=-np.inf); H,W=a.shape
        return np.max([p[i:i+H,j:j+W] for i in range(3) for j in range(3)],0)>0.1
    def step(x,rng,fr=0.5):
        C,H,W=x.shape; p=np.pad(x,((0,0),(1,1),(1,1)))
        sh=[p[:,i:i+H,j:j+W] for i in range(3) for j in range(3)]
        per=np.stack([sum(K[k,i,j]*sh[3*i+j] for i in range(3) for j in range(3) if K[k,i,j]!=0) for k in range(3)],1).reshape(3*C,H*W)
        dx=(w2@np.maximum(w1@per+b1,0)).reshape(C,H,W)
        if fr<1: dx*= (rng.random((H,W))<=fr)
        n=x+dx; return (n*(alive(x[3])&alive(n[3]))).astype(np.float32)
    return step
def target(name):
    t=np.asarray(Image.open(f"neural-cellular-automata/data/{name}.png"),np.float32)/255; t[...,:3]*=t[...,3:]; return t.transpose(2,0,1)
def circle(rng,S=28):
    x=np.linspace(-1,1,S)[None,:]; y=np.linspace(-1,1,S)[:,None]; c=rng.uniform(-.5,.5,2); r=rng.uniform(.1,.4)
    return (((x-c[0])/r)**2+((y-c[1])/r)**2<1).astype(np.float32)
def evaluate(path,filt,name,seeds=5,T=1000,dmg_t=300):
    sd=load_pt(path); step=make_step(sd,filt); tg=target(name); out=[]
    for sdv in range(seeds):
        rng=np.random.default_rng(sdv); x=np.zeros((16,28,28),np.float32); x[3:,14,14]=1; mse=[]
        for t in range(1,T+1):
            if t==dmg_t+1: x*=1-circle(rng)
            x=step(x,rng); mse.append(float(((x[:4]-tg)**2).mean()))
        mse=np.array(mse); pre=mse[dmg_t-1]; post=mse[dmg_t:]
        rec=next((i for i,v in enumerate(post) if v<=max(2*pre,1e-3)),None)
        out.append(dict(mse200=mse[199],mse1000=mse[-1],peak_after_damage=post.max(),recover_steps=rec))
    return out
t0=time.time()
for rel,filt in [("chest/L2/chest_sobel.pt","sobel"),("chest/L2/chest_sobel_damage.pt","sobel"),("chest/L2/chest_laplacian.pt","laplacian")]:
    r=evaluate("neural-cellular-automata/models/"+rel,filt,"chest")
    agg={k:(np.mean([o[k] for o in r if o[k] is not None]),np.std([o[k] for o in r if o[k] is not None]), sum(o[k] is None for o in r)) for k in r[0]}
    print(rel, {k:f"{m:.4f}±{s:.4f}"+(f" (failed {f})" if f else "") for k,(m,s,f) in agg.items()})
print("time %.1fs for 3 models x 5 seeds x 1000 steps"%(time.time()-t0))
