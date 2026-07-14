"""make_gif.py — eyeball GIF of the 128^3 Re_lambda~37 flow: |omega| + u_x mid-plane.
Vorticity via spectral curl. Downsampled for a small viewable file.
Usage: KMP_DUPLICATE_LIB_OK=TRUE python scripts/make_gif.py <seed_dir> <out.gif> [stride]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solver  # noqa env guard
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, glob

seed_dir = sys.argv[1]
out = Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
small = "--small" in sys.argv

N = 128
k = np.fft.fftfreq(N, 1.0/N)
KX, KY, KZ = [torch.tensor(a) for a in np.meshgrid(k, k, k, indexing="ij")]

def om_ux(u):  # u [3,N,N,N] -> |omega| midplane, u_x midplane (downsampled if small)
    uh = [torch.fft.fftn(u[c].double()) for c in range(3)]
    wx = torch.fft.ifftn(1j*KZ*uh[1] - 1j*KY*uh[2]).real
    wy = torch.fft.ifftn(1j*KX*uh[2] - 1j*KZ*uh[0]).real
    wz = torch.fft.ifftn(1j*KY*uh[0] - 1j*KX*uh[1]).real
    om = torch.sqrt(wx*wx + wy*wy + wz*wz)[N//2].numpy()
    ux = u[0][N//2].double().numpy()
    if small:
        om = om[::2, ::2]; ux = ux[::2, ::2]
    return om, ux

fs = sorted(glob.glob(f"{seed_dir}/frame*.pt"))[::stride]
oms, uxs, ts = [], [], []
for i, f in enumerate(fs):
    d = torch.load(f, map_location="cpu", weights_only=False)
    o, x = om_ux(d["u"]); oms.append(o); uxs.append(x); ts.append(float(d["t"]))
    if (i+1) % 20 == 0: print(f"  {i+1}/{len(fs)}")
oms = np.array(oms); uxs = np.array(uxs); ts = np.array(ts); t0 = ts[0]
om_max = float(np.percentile(oms, 99.5)); ux_max = float(np.abs(uxs).max())
print(f"{len(fs)} frames  |omega| clim [0,{om_max:.1f}]  ux +-{ux_max:.2f}")

figsize = (8, 3.9) if small else (11, 5.0)
fig, ax = plt.subplots(1, 2, figsize=figsize)
im0 = ax[0].imshow(oms[0], cmap="inferno", origin="lower", vmin=0, vmax=om_max)
ax[0].set_title(r"$|\omega|$", fontsize=10); ax[0].set_xticks([]); ax[0].set_yticks([])
im1 = ax[1].imshow(uxs[0], cmap="RdBu_r", origin="lower", vmin=-ux_max, vmax=ux_max)
ax[1].set_title(r"$u_x$", fontsize=10); ax[1].set_xticks([]); ax[1].set_yticks([])
sup = fig.suptitle("", fontsize=10); fig.tight_layout()
span = (ts[-1]-t0)

def update(i):
    im0.set_data(oms[i]); im1.set_data(uxs[i])
    sup.set_text(f"128$^3$ fp64 Re$_\\lambda$~37 k_f=4  t={ts[i]-t0:.1f}/{span:.0f} t.u.")
    return im0, im1, sup

from matplotlib.animation import FuncAnimation, PillowWriter
anim = FuncAnimation(fig, update, frames=len(fs), blit=False)
dpi = 55 if small else 80
anim.save(out, writer=PillowWriter(fps=12), dpi=dpi); plt.close(fig)
print(f"saved {out} ({out.stat().st_size/1e6:.2f} MB)")
