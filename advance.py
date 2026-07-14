"""advance.py — the ONE simple entry point for adversarial ML.

You give us a velocity field x (your original frame, OR your frame with an adversarial
perturbation already added — any array), we return the true DNS next frame.

We do NOT require the perturbation to be pre-sanitized: any x is first projected onto the
legal incompressible-DNS manifold (Leray projection + 2/3 dealiasing), so whatever
adversarial modification you applied, the input we actually step is a valid DNS state. Then
we advance the real solver one frame and return it.

Simplest possible use:

    from advance import advance
    x_next = advance(x)                 # x: (3,128,128,128) numpy/torch, any real dtype
    # x_next: (3,128,128,128) numpy float32 — the true next DNS frame

Optional knobs:
    advance(x, dt=None, seed=0, return_info=False)
      dt          : sub-step size; None = CFL-chosen (default, recommended)
      seed        : forcing realization (matches dataset seeds 0..7 -> ou_seed 1000..1007)
      frame_dt    : advance by this much sim-time (default one dataset frame, 0.30)
      return_info : also return a dict {div_residual, k_max_eta, K, legal}

Helper if you want us to add the perturbation instead of you:

    from advance import advance, legal_perturb
    x_adv = legal_perturb(x, amp=0.1, seed=0)   # a legal adversarial example
    x_next = advance(x_adv)
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import solver  # noqa: F401  env guard (Windows OpenMP)
import numpy as np
import torch

from solver import SpectralGrid, PseudoSpectralSolver
from solver.config import SolverConfig, ForcingConfig
from solver.operators import leray_project_, dealias_

# dataset config (must match generate_dataset.py)
N, NU, KF, SIGMA2, OU_TAU, FRAME_DT = 128, 0.006, 4.0, 0.16, 1.0, 0.30
_FFT = (-3, -2, -1)
_GRID = None


def _grid() -> SpectralGrid:
    global _GRID
    if _GRID is None:
        _GRID = SpectralGrid(N, "cuda" if torch.cuda.is_available() else "cpu", "fp64")
    return _GRID


def _to_uhat(x, grid) -> torch.Tensor:
    """Any (3,N,N,N) field -> solenoidal, dealiased spectral field (legal DNS state)."""
    t = torch.as_tensor(np.asarray(x)) if not torch.is_tensor(x) else x
    t = t.to(grid.device, grid.rdtype)
    assert t.shape == (3, N, N, N), f"x must be (3,{N},{N},{N}), got {tuple(t.shape)}"
    uh = torch.fft.rfftn(t, dim=_FFT)
    dealias_(uh, grid)
    leray_project_(uh, grid)          # project onto the incompressible manifold
    return uh


def legal_perturb(x, amp: float = 0.10, seed: int = 0, k_lo: float = 1.0, k_hi: float = 8.0):
    """Add a physically-legal adversarial perturbation to x (rel rms budget `amp`).
    Returns a (3,N,N,N) numpy float32 field that is a legal DNS state."""
    grid = _grid()
    uh = _to_uhat(x, grid)
    import math
    g = torch.Generator(device="cpu").manual_seed(seed)
    pert = (torch.randn(uh.shape, generator=g, dtype=grid.rdtype)
            + 1j * torch.randn(uh.shape, generator=g, dtype=grid.rdtype)).to(grid.device, grid.cdtype)
    kmag = grid.k2.sqrt()
    pert *= ((kmag >= k_lo) & (kmag <= k_hi)).unsqueeze(0)
    dealias_(pert, grid); leray_project_(pert, grid)
    base = math.sqrt(2 * grid.kinetic_energy(uh) / 3.0)
    pr = math.sqrt(2 * grid.kinetic_energy(pert) / 3.0) + 1e-30
    pert *= (amp * base / pr)
    out = uh + pert
    dealias_(out, grid); leray_project_(out, grid)
    u = torch.fft.irfftn(out, s=(N, N, N), dim=_FFT)
    return u.to(torch.float32).cpu().numpy()


def advance(x, dt: float | None = None, seed: int = 0, frame_dt: float = FRAME_DT,
            return_info: bool = False):
    """Return the true DNS next frame from x (original OR adversarially perturbed).

    x           : (3,128,128,128) array (numpy or torch), any real dtype.
    Returns     : (3,128,128,128) numpy float32 next frame.
    return_info : also return a dict with div_residual / k_max_eta / K / legal.
    """
    grid = _grid()
    uh = _to_uhat(x, grid)                 # sanitize any input to a legal DNS state
    scfg = SolverConfig(N=N, nu=NU, dtype="fp64",
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        scheme="rk3", cfl=0.4, dt_max=0.01,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                                              ou_sigma2=SIGMA2, ou_seed=1000 + seed))
    s = PseudoSpectralSolver(scfg, uh)
    t_target = frame_dt
    while s.t < t_target:
        s.step(dt if dt is not None else s.suggest_dt(t_target))
    u_next = torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=_FFT).to(torch.float32).cpu().numpy()
    if not return_info:
        return u_next
    # legality of the returned frame
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    div = kx*s.u_hat[0] + ky*s.u_hat[1] + kz*s.u_hat[2]
    grad2 = float((grid.k2 * (s.u_hat.abs()**2).sum(0)).sum())
    div_res = float((div.abs()**2).sum().sqrt() / (grad2**0.5 + 1e-30))
    eps = grid.dissipation(s.u_hat, NU); eta = (NU**3/eps)**0.25
    keta = grid.k_max_resolved * eta
    info = {"div_residual": div_res, "k_max_eta": keta,
            "K": grid.kinetic_energy(s.u_hat), "legal": div_res < 1e-6 and keta >= 1.5}
    return u_next, info


if __name__ == "__main__":
    # tiny demo on a MATURED field (spin the solver up so the spectrum is stationary,
    # like a real dataset frame — a cold random IC has k_maxeta<1.5 and is NOT a valid
    # dataset state, so we never demo on one).
    grid = _grid()
    from solver.initial_conditions import random_solenoidal
    scfg = SolverConfig(N=N, nu=NU, dtype="fp64",
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        scheme="rk3", cfl=0.4, dt_max=0.01,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                                              ou_sigma2=SIGMA2, ou_seed=1000))
    s = PseudoSpectralSolver(scfg, random_solenoidal(grid, seed=0, k_p=4.0, u_rms=0.7))
    while s.t < 15.0:
        s.step(s.suggest_dt(15.0))
    x = torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=_FFT).cpu().numpy()   # a matured frame
    xn, info = advance(x, return_info=True)
    print("advance() demo (matured frame):", xn.shape, xn.dtype, "| next-frame legal:", info["legal"],
          f"div={info['div_residual']:.1e} k_maxeta={info['k_max_eta']:.3f}")
