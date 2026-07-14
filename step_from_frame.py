"""step_from_frame.py — the adversarial core.

Scenario: an adversary takes one of our DNS frames and adds a *physically-legal*
perturbation (an "adversarial example" that stays on the incompressible-DNS manifold).
We must be able to (a) load ANY frame, (b) apply a legal perturbation, (c) advance the
solver ONE step to compute the true next frame, and (d) confirm the perturbed input and
its evolution remain valid DNS (incompressible + resolved).

Two public entry points:

  load_frame_as_uhat(path)          -> u_hat on the solver grid (any saved frame)
  legal_perturbation(u_hat, ...)    -> perturbed u_hat, GUARANTEED solenoidal+dealiased
  step_one(u_hat, ...)              -> (u_hat_next, dt)   true DNS next frame
  input_legality(u_hat)             -> dict: divergence residual, k_maxeta, spectrum tail

"Physically-legal perturbation" = we add a small solenoidal, band-limited velocity field
(so it is a valid incompressible flow), then Leray-project + dealias the sum so the result
is EXACTLY a legal DNS state (div u = 0 to machine precision, no aliased modes). The
perturbation is otherwise arbitrary (adversary-chosen direction/magnitude), which is the
adversarial-ML setting: the input is attacker-modified but still a legal DNS field, and we
return the physically-correct next frame the solver produces from it.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path

import solver  # noqa: F401  env guard
import torch

from solver import SpectralGrid, PseudoSpectralSolver
from solver.config import SolverConfig, ForcingConfig
from solver.operators import leray_project_, dealias_, curl_hat

N, NU, KF, SIGMA2, OU_TAU = 128, 0.006, 4.0, 0.16, 1.0
_FFT_DIMS = (-3, -2, -1)


def _grid() -> SpectralGrid:
    return SpectralGrid(N, "cuda", "fp64")


def load_frame_as_uhat(path, grid: SpectralGrid | None = None) -> torch.Tensor:
    """Load a saved frame's physical velocity -> solenoidal, dealiased u_hat."""
    grid = grid or _grid()
    d = torch.load(path, map_location="cpu", weights_only=False)
    u = d["u"].to(grid.device, grid.rdtype)              # [3,N,N,N] fp64
    u_hat = torch.fft.rfftn(u, dim=_FFT_DIMS)
    dealias_(u_hat, grid)
    leray_project_(u_hat, grid)                          # ensure exactly solenoidal
    return u_hat


def legal_perturbation(u_hat: torch.Tensor, grid: SpectralGrid,
                       amp: float = 0.10, k_lo: float = 1.0, k_hi: float = 8.0,
                       seed: int = 0) -> torch.Tensor:
    """Add an adversary-chosen but PHYSICALLY-LEGAL perturbation.

    Build a random band-limited velocity field, make it solenoidal, scale it to `amp`
    times the base field's rms, add it, then Leray-project + dealias the SUM so the
    output is exactly a legal DNS state (div=0 to machine precision, no aliased modes).
    `amp` is the relative perturbation strength (adversary budget).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # random complex field in a spherical shell k_lo <= |k| <= k_hi
    re = torch.randn(u_hat.shape, generator=g, dtype=grid.rdtype)
    im = torch.randn(u_hat.shape, generator=g, dtype=grid.rdtype)
    pert = (re + 1j * im).to(grid.device, grid.cdtype)
    kmag = grid.k2.sqrt()
    shell = ((kmag >= k_lo) & (kmag <= k_hi)).unsqueeze(0)
    pert *= shell
    dealias_(pert, grid)
    leray_project_(pert, grid)                           # make perturbation solenoidal
    # scale to amp * rms(base)
    base_rms = math.sqrt(2.0 * grid.kinetic_energy(u_hat) / 3.0)
    pert_rms = math.sqrt(2.0 * grid.kinetic_energy(pert) / 3.0) + 1e-30
    pert *= (amp * base_rms / pert_rms)
    out = u_hat + pert
    dealias_(out, grid)
    leray_project_(out, grid)                            # sum is exactly legal DNS
    return out


def input_legality(u_hat: torch.Tensor, grid: SpectralGrid) -> dict:
    """Is this a legal DNS input? divergence residual + resolution + spectrum tail."""
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    div_hat = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    # relative divergence: ||div u|| / ||grad u||
    grad2 = (grid.k2 * (u_hat.abs() ** 2).sum(0)).sum()
    div2 = (div_hat.abs() ** 2).sum()
    div_res = float((div2 / (grad2 + 1e-30)).sqrt())
    eps = grid.dissipation(u_hat, NU)
    eta = (NU**3 / eps) ** 0.25
    return {"div_residual": div_res, "k_max_eta": grid.k_max_resolved * eta,
            "K": grid.kinetic_energy(u_hat), "eps": eps}


def step_one(u_hat: torch.Tensor, grid: SpectralGrid, seed: int = 0,
             dt: float | None = None) -> tuple[torch.Tensor, float]:
    """Advance the given field ONE DNS step (same solver as production). Returns
    (u_hat_next, dt). Builds a solver seeded to the config's forcing so the step is
    the physically-correct evolution of THIS input frame."""
    scfg = SolverConfig(N=N, nu=NU, dtype="fp64", device="cuda", scheme="rk3",
                        cfl=0.4, dt_max=0.01,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                                              ou_sigma2=SIGMA2, ou_seed=1000 + seed))
    s = PseudoSpectralSolver(scfg, u_hat)
    dt = dt if dt is not None else s.suggest_dt()
    s.step(dt)
    return s.u_hat.clone(), dt


def _demo(frame_path: str, amp: float, seed: int):
    grid = _grid()
    print(f"=== adversarial step demo: {frame_path} (perturb amp={amp}) ===")
    u0 = load_frame_as_uhat(frame_path, grid)
    L0 = input_legality(u0, grid)
    print(f"[base frame]      div_res={L0['div_residual']:.2e}  k_maxeta={L0['k_max_eta']:.3f}  K={L0['K']:.4f}")

    up = legal_perturbation(u0, grid, amp=amp, seed=seed)
    Lp = input_legality(up, grid)
    d_rms = math.sqrt(2 * grid.kinetic_energy(up - u0) / 3.0)
    base_rms = math.sqrt(2 * grid.kinetic_energy(u0) / 3.0)
    print(f"[perturbed input] div_res={Lp['div_residual']:.2e}  k_maxeta={Lp['k_max_eta']:.3f}  K={Lp['K']:.4f}"
          f"  |delta|/|u|={d_rms/base_rms:.3f}")

    un, dt = step_one(up, grid, seed=seed)
    Ln = input_legality(un, grid)
    print(f"[next frame]      div_res={Ln['div_residual']:.2e}  k_maxeta={Ln['k_max_eta']:.3f}  K={Ln['K']:.4f}  (dt={dt:.4f})")

    ok = (Lp["div_residual"] < 1e-6 and Ln["div_residual"] < 1e-6
          and Lp["k_max_eta"] >= 1.5 and Ln["k_max_eta"] >= 1.5)
    print(f"=> perturbed input + next frame are LEGAL DNS (div<1e-6, Class I): {'YES' if ok else 'NO'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="path to a saved frame*.pt")
    ap.add_argument("--amp", type=float, default=0.10, help="adversarial perturbation strength (rel rms)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _demo(args.frame, args.amp, args.seed)
