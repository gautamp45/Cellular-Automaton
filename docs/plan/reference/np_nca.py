"""Numpy reference NCA: loads .pt without torch, runs the same update as model.py."""
import zipfile, pickle, sys, glob
import numpy as np
from PIL import Image

def load_pt(path):
    z = zipfile.ZipFile(path); root = z.namelist()[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *rest):
        flat = storage
        return np.lib.stride_tricks.as_strided(flat[offset:], shape=size, strides=[s*4 for s in stride]).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name == "OrderedDict": import collections; return collections.OrderedDict
            return lambda *a, **k: None
        def persistent_load(self, pid):
            key = pid[2]
            return np.frombuffer(z.read(f"{root}/data/{key}"), dtype="<f4")
    return U(z.open(f"{root}/data.pkl")).load()

FILTERS = {"sobel": ([[-1,0,1],[-2,0,2],[-1,0,1]], 8.), "scharr": ([[-3,0,3],[-10,0,10],[-3,0,3]], 16.),
           "gaussian": ([[1,2,1],[2,4,2],[1,2,1]], 16.), "laplacian": ([[0,1,0],[1,-4,1],[0,1,0]], 8.),
           "mean": ([[1,1,1],[1,1,1],[1,1,1]], 9.)}

def corr3(x, k):  # x: (C,H,W) zero-padded 3x3 cross-correlation
    p = np.pad(x, ((0,0),(1,1),(1,1))); H, W = x.shape[1:]
    return sum(k[i][j] * p[:, i:i+H, j:j+W] for i in range(3) for j in range(3))

def alive(x):
    a = np.pad(x[3], 1, constant_values=-np.inf); H, W = x.shape[1:]
    return np.max([a[i:i+H, j:j+W] for i in range(3) for j in range(3)], axis=0) > 0.1

def step(x, sd, filt, fire_rate=1.0, rng=None):
    f, s = FILTERS[filt]; f = np.array(f, np.float32)
    kid = np.zeros((3,3), np.float32); kid[1,1] = 1
    per = np.stack([corr3(x, kid), corr3(x, f/s), corr3(x, f.T/s)], axis=1).reshape(48, *x.shape[1:])  # index 3c+k
    w1 = sd["update_module.0.weight"][:, :, 0, 0]; b1 = sd["update_module.0.bias"]; w2 = sd["update_module.2.weight"][:, :, 0, 0]
    h = np.maximum(np.einsum("oc,chw->ohw", w1, per) + b1[:, None, None], 0)
    dx = np.einsum("oc,chw->ohw", w2, h)
    if fire_rate < 1.0: dx = dx * (rng.random(x.shape[1:]) <= fire_rate)
    new = x + dx
    return new * (alive(x) & alive(new))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for p in sorted(glob.glob("neural-cellular-automata/models/**/*.pt", recursive=True)):
        target_name = p.split("/")[2]; filt = p.split("_")[-1].replace(".pt","") if "damage" not in p else "sobel"
        filt = [f for f in FILTERS if f in p.split("/")[-1]][0]
        sd = load_pt(p)
        t = np.asarray(Image.open(f"neural-cellular-automata/data/{target_name}.png").resize((28,28), Image.LANCZOS), np.float32)/255
        t[..., :3] *= t[..., 3:]; t = t.transpose(2,0,1)
        x = np.zeros((16,28,28), np.float32); x[3:,14,14] = 1
        losses = {}
        for i in range(1, 201):
            x = step(x, sd, filt, 0.5, rng)
            if i in (50, 100, 200): losses[i] = float(((x[:4]-t)**2).mean())
        print(f"{p.split('models/')[1]:40s} filt={filt:9s} L2@50/100/200 = " + " ".join(f"{v:.4f}" for v in losses.values()) + f"  alive={alive(x).mean():.2f}")
