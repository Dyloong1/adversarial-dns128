"""advance.py — the adversarial-DNS service interface.

We provide the DNS "next frame" service; the *attacker's* developer chooses the threat
model. Two explicit paths, no hidden preprocessing:

  advance_raw(x)        — step x AS GIVEN. Does NOT sanitize. Checks legality and, if x is
                          not a legal DNS state (compressible / under-resolved), RAISES (or
                          warns) instead of silently "fixing" it. Use this when the attacker
                          is constrained to produce legal perturbations — the model's
                          robustness is measured on exactly the attacker's field.

  advance_projected(x)  — project x onto the legal incompressible-DNS manifold (Leray +
                          dealias) and THEN step. Use this when your pipeline is allowed to
                          purify inputs. NOTE for adversarial papers: this projection is an
                          input-purification defense — it removes the compressible/aliased
                          part of the attacker's perturbation, so report it as part of the
                          pipeline, not as a neutral format step.

  advance(x, project=False)  — unified entry; project=False -> raw, True -> projected.

Helpers (let the attacker decide):
  input_legality(x)     — {div_residual, k_max_eta, K, compressible_fraction, legal}
  project_to_manifold(x)— return the Leray-projected, dealiased legal field (no stepping)
  legal_perturb(x, amp) — build a LEGAL adversarial example for you (solenoidal, resolved)

x is (3,128,128,128) (channels u,v,w), numpy or torch, any real dtype. Returns numpy float32.
"""
from __future__ import annotations
from pathlib import Path
import sys, math, warnings
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
# legality thresholds. The dataset is stored fp32, so a legal frame round-tripped through
# fp32 carries ~1e-7 numerical noise in the divergence / dealias-band residuals — these
# tolerances are set to fp32 storage precision (with margin), NOT fp64 machine epsilon, so a
# genuinely legal (fp32) field is accepted while a real compressible/aliased attack (which is
# orders of magnitude larger, ~1e-1) is still refused.
DIV_TOL = 1e-5        # relative divergence residual to count as incompressible (fp32-safe)
ALIAS_TOL = 1e-5      # aliased-band energy fraction (fp32-safe)
KETA_MIN = 1.5        # Class I resolution
_GRID = None


def _grid() -> SpectralGrid:
    global _GRID
    if _GRID is None:
        _GRID = SpectralGrid(N, "cuda" if torch.cuda.is_available() else "cpu", "fp64")
    return _GRID


def _fft(x, grid) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x)) if not torch.is_tensor(x) else x
    t = t.to(grid.device, grid.rdtype)
    assert t.shape == (3, N, N, N), f"x must be (3,{N},{N},{N}), got {tuple(t.shape)}"
    return torch.fft.rfftn(t, dim=_FFT)


def _legality_of_uhat(uh, grid) -> dict:
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    div = kx*uh[0] + ky*uh[1] + kz*uh[2]
    grad2 = float((grid.k2 * (uh.abs()**2).sum(0)).sum()) + 1e-30
    div_res = float((div.abs()**2).sum().sqrt() / grad2**0.5)
    # compressible energy fraction = ||dilatational|| / ||u||
    total = float((uh.abs()**2).sum()) + 1e-30
    dil = float((div.abs()**2 * grid.inv_k2).sum())     # |k.u|^2/k^2 = dilatational energy
    comp_frac = (dil / total) ** 0.5
    # resolution: does energy respect the 2/3 dealias band? aliased energy above k_cut
    aliased = float((uh.abs()**2 * (~grid.dealias_mask)).sum())
    alias_frac = (aliased / total) ** 0.5
    eps = grid.dissipation(uh, NU); eta = (NU**3/max(eps, 1e-30))**0.25
    keta = grid.k_max_resolved * eta
    legal = (div_res <= DIV_TOL) and (keta >= KETA_MIN) and (alias_frac < ALIAS_TOL)
    return {"div_residual": div_res, "k_max_eta": keta, "K": grid.kinetic_energy(uh),
            "compressible_fraction": comp_frac, "aliased_fraction": alias_frac, "legal": legal}


def input_legality(x) -> dict:
    """Is x a legal DNS input? Report divergence, resolution, compressible/aliased fractions."""
    grid = _grid()
    return _legality_of_uhat(_fft(x, grid), grid)


def project_to_manifold(x):
    """Return x projected onto the legal incompressible-DNS manifold (Leray + dealias). No step."""
    grid = _grid()
    uh = _fft(x, grid)
    dealias_(uh, grid); leray_project_(uh, grid)
    return torch.fft.irfftn(uh, s=(N, N, N), dim=_FFT).to(torch.float32).cpu().numpy()


def _step_uhat(uh, seed, frame_dt, dt):
    scfg = SolverConfig(N=N, nu=NU, dtype="fp64",
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        scheme="rk3", cfl=0.4, dt_max=0.01,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                                              ou_sigma2=SIGMA2, ou_seed=1000 + seed))
    s = PseudoSpectralSolver(scfg, uh)
    while s.t < frame_dt:
        s.step(dt if dt is not None else s.suggest_dt(frame_dt))
    return s.u_hat


def advance_raw(x, seed: int = 0, frame_dt: float = FRAME_DT, dt: float | None = None,
                on_illegal: str = "raise", return_info: bool = False):
    """Step x AS GIVEN (no sanitization). If x is not a legal DNS state, `on_illegal`
    decides: 'raise' (default) / 'warn' / 'ignore'. This measures model robustness on
    exactly the attacker-provided field — the attacker is responsible for legality.

    NOTE: the spectral solver still needs a solenoidal input to be well-posed. advance_raw
    does NOT project, but a grossly compressible input will produce a physically meaningless
    step — which is why we refuse it by default rather than silently Leray-projecting."""
    grid = _grid()
    uh = _fft(x, grid)
    L = _legality_of_uhat(uh, grid)
    if not L["legal"]:
        msg = (f"advance_raw: input is NOT a legal DNS state "
               f"(div_residual={L['div_residual']:.2e}, k_maxeta={L['k_max_eta']:.3f}, "
               f"compressible_fraction={L['compressible_fraction']:.2e}, "
               f"aliased_fraction={L['aliased_fraction']:.2e}). "
               f"The attacker must supply a legal (incompressible, Class-I, dealiased) field, "
               f"or call advance_projected()/project_to_manifold() explicitly.")
        if on_illegal == "raise":
            raise ValueError(msg)
        elif on_illegal == "warn":
            warnings.warn(msg)
        # 'ignore' -> proceed anyway
    uh_next = _step_uhat(uh, seed, frame_dt, dt)
    out = torch.fft.irfftn(uh_next, s=(N, N, N), dim=_FFT).to(torch.float32).cpu().numpy()
    if return_info:
        return out, {"input_legality": L, "next": _legality_of_uhat(uh_next, grid)}
    return out


def advance_projected(x, seed: int = 0, frame_dt: float = FRAME_DT, dt: float | None = None,
                      return_info: bool = False):
    """Project x onto the legal manifold (Leray + dealias) THEN step. This projection is an
    input-purification step — report it as part of your pipeline for adversarial papers."""
    grid = _grid()
    uh = _fft(x, grid)
    dealias_(uh, grid); leray_project_(uh, grid)
    uh_next = _step_uhat(uh, seed, frame_dt, dt)
    out = torch.fft.irfftn(uh_next, s=(N, N, N), dim=_FFT).to(torch.float32).cpu().numpy()
    if return_info:
        return out, {"next": _legality_of_uhat(uh_next, grid)}
    return out


def advance(x, project: bool = False, seed: int = 0, frame_dt: float = FRAME_DT,
            dt: float | None = None, return_info: bool = False, on_illegal: str = "raise"):
    """Unified entry. project=False -> advance_raw (default, no purification);
    project=True -> advance_projected."""
    if project:
        return advance_projected(x, seed=seed, frame_dt=frame_dt, dt=dt, return_info=return_info)
    return advance_raw(x, seed=seed, frame_dt=frame_dt, dt=dt, return_info=return_info,
                       on_illegal=on_illegal)


def legal_perturb(x, amp: float = 0.10, seed: int = 0, k_lo: float = 1.0, k_hi: float = 8.0):
    """Build a physically-legal adversarial example: add a solenoidal, band-limited field
    of relative rms `amp`, keep the sum on the legal manifold. Returns (3,N,N,N) float32."""
    grid = _grid()
    uh = _fft(x, grid)
    dealias_(uh, grid); leray_project_(uh, grid)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pert = (torch.randn(uh.shape, generator=g, dtype=grid.rdtype)
            + 1j*torch.randn(uh.shape, generator=g, dtype=grid.rdtype)).to(grid.device, grid.cdtype)
    kmag = grid.k2.sqrt()
    pert *= ((kmag >= k_lo) & (kmag <= k_hi)).unsqueeze(0)
    dealias_(pert, grid); leray_project_(pert, grid)
    base = math.sqrt(2*grid.kinetic_energy(uh)/3.0)
    pr = math.sqrt(2*grid.kinetic_energy(pert)/3.0) + 1e-30
    pert *= (amp*base/pr)
    out = uh + pert
    dealias_(out, grid); leray_project_(out, grid)
    return torch.fft.irfftn(out, s=(N, N, N), dim=_FFT).to(torch.float32).cpu().numpy()


if __name__ == "__main__":
    # demo on a matured legal frame
    grid = _grid()
    from solver.initial_conditions import random_solenoidal
    s = PseudoSpectralSolver(
        SolverConfig(N=N, nu=NU, dtype="fp64",
                     device="cuda" if torch.cuda.is_available() else "cpu",
                     scheme="rk3", cfl=0.4, dt_max=0.01,
                     forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                                           ou_sigma2=SIGMA2, ou_seed=1000)),
        random_solenoidal(grid, seed=0, k_p=4.0, u_rms=0.7))
    while s.t < 15.0:
        s.step(s.suggest_dt(15.0))
    x = torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=_FFT).cpu().numpy()
    x_adv = legal_perturb(x, amp=0.10, seed=1)          # a legal adversarial example
    xn, info = advance_raw(x_adv, return_info=True)
    print("advance_raw(legal adversarial x):", xn.shape, "| input legal:", info["input_legality"]["legal"],
          "| next legal:", info["next"]["legal"])
