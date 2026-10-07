import json, re, numpy as np, time
import sys
# Download first: curl -sSO https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'animals.json')); orb=[a for a in d if a.get('code')=='O2u'][0]
def decode(rle):
    rows=[[]]; n=''; i=0; s=rle.rstrip('!')
    while i<len(s):
        ch=s[i]
        if ch.isdigit(): n+=ch; i+=1; continue
        k=int(n or 1); n=''
        if ch=='$': rows+= [[] for _ in range(k)]; i+=1; continue
        if ch=='.': rows[-1]+=[0]*k; i+=1; continue
        if 'p'<=ch<='y': v=(ord(ch)-ord('p')+1)*24+(ord(s[i+1])-ord('A')+1); i+=2
        else: v=ord(ch)-ord('A')+1; i+=1
        rows[-1]+=[v]*k
    w=max(map(len,rows)); return np.array([r+[0]*(w-len(r)) for r in rows],float)/255
cells=decode(orb['cells']); print("orbium shape",cells.shape,"max",cells.max().round(3),"mass",cells.sum().round(3))
P=orb['params']; R,T,m,s=P['R'],P['T'],P['m'],P['s']
N=128
def kernel(R,N,b=(1,)):
    y,x=np.ogrid[-N//2:N//2,-N//2:N//2]; r=np.sqrt(x*x+y*y)/R; B=len(b)
    Br=B*r; idx=np.minimum(Br.astype(int),B-1); f=Br%1
    with np.errstate(divide='ignore',over='ignore'):
        core=np.where((f>0)&(f<1), np.exp(4-1/(f*(1-f))),0)
    K=np.where(r<1, np.array(b)[idx]*core, 0); K/=K.sum(); return np.fft.fft2(np.fft.ifftshift(K))
fK=kernel(R,N)
G=lambda u: 2*np.exp(-((u-m)**2)/(2*s*s))-1
A=np.zeros((N,N)); A[40:40+cells.shape[0],40:40+cells.shape[1]]=cells
def com(A):
    # circular centroid
    ang=2*np.pi*np.arange(N)/N; w=A.sum()
    cy=np.angle((A.sum(1)*np.exp(1j*ang)).sum())%(2*np.pi)*N/(2*np.pi); cx=np.angle((A.sum(0)*np.exp(1j*ang)).sum())%(2*np.pi)*N/(2*np.pi); return cy,cx
c0=com(A); masses=[]; t0=time.time()
for t in range(1,501):
    U=np.real(np.fft.ifft2(fK*np.fft.fft2(A))); A=np.clip(A+G(U)/T,0,1)
    if t in (1,100,200,300,400,500): masses.append((t,round(float(A.sum()),3),tuple(round(v,1) for v in com(A))))
print("time %.2fs"%(time.time()-t0)); print("t, mass, centroid:", masses)
# Life as special case: R=1 3x3 box kernel incl centre? Chan: Life = Lenia with rectangular kernel, step growth
