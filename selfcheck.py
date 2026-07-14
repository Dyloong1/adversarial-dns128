"""selfcheck.py — one-shot self-check for the self-contained 128^3 adversarial-DNS package.

Runs the whole pipeline on a tiny scale and asserts each stage works:
  1. solver builds + steps on 128^3 fp64
  2. generate a few frames (matured, Class I)
  3. adversarial core: legal perturbation of a frame -> step -> next frame stays legal DNS
  4. A+D eval runs and D-group passes (A10 needs many frames; checked separately at full scale)

Usage:  KMP_DUPLICATE_LIB_OK=TRUE python selfcheck.py
Exit 0 = all green.
"""
from __future__ import annotations
import sys, math, tempfile, shutil
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import solver  # noqa env guard
import torch

from solver import SpectralGrid, PseudoSpectralSolver
from solver.config import SolverConfig, ForcingConfig
from solver.initial_conditions import random_solenoidal

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def main():
    print("=== adversarial_dns128 self-check ===")
    # 1. solver on 128^3
    grid = SpectralGrid(128, "cuda", "fp64")
    scfg = SolverConfig(N=128, nu=0.006, dtype="fp64", device="cuda", scheme="rk3",
                        cfl=0.4, dt_max=0.01,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=4.0, ou_tau=1.0,
                                              ou_sigma2=0.16, ou_seed=1000))
    u_hat0 = random_solenoidal(grid, seed=0, k_p=4.0, u_rms=0.7)
    s = PseudoSpectralSolver(scfg, u_hat0)
    s.step(s.suggest_dt())
    check("solver builds + steps on 128^3 fp64", s.grid.N == 128 and math.isfinite(s.scalars()["K"]))

    # 2/3. adversarial core on a matured frame
    from step_from_frame import legal_perturbation, step_one, input_legality, load_frame_as_uhat
    # matured frame: quick spin-up
    while s.t < 15.0:
        s.step(s.suggest_dt(15.0))
    uh = s.u_hat.clone()
    L0 = input_legality(uh, grid)
    check("base frame Class I + incompressible", L0["k_max_eta"] >= 1.5 and L0["div_residual"] < 1e-6,
          f"k_maxeta={L0['k_max_eta']:.3f} div={L0['div_residual']:.1e}")
    up = legal_perturbation(uh, grid, amp=0.10, seed=1)
    Lp = input_legality(up, grid)
    d_rms = math.sqrt(2*grid.kinetic_energy(up - uh)/3.0)
    base_rms = math.sqrt(2*grid.kinetic_energy(uh)/3.0)
    check("legal perturbation stays incompressible + Class I", Lp["div_residual"] < 1e-6 and Lp["k_max_eta"] >= 1.5,
          f"|delta|/|u|={d_rms/base_rms:.3f} div={Lp['div_residual']:.1e}")
    un, dt = step_one(up, grid, seed=0)
    Ln = input_legality(un, grid)
    check("stepped next frame from perturbed input stays legal DNS", Ln["div_residual"] < 1e-6 and Ln["k_max_eta"] >= 1.5,
          f"k_maxeta={Ln['k_max_eta']:.3f} div={Ln['div_residual']:.1e} dt={dt:.4f}")

    # 4. D-group on this frame (fast, decisive)
    from eval.eval_ad import d_group
    D = d_group(uh, grid, seed=0)
    check("D1 divergence <=1e-6", D["D1_div"] <= 1e-6, f"{D['D1_div']:.1e}")
    check("D2 NS residual <=1e-2", D["D2_res"] <= 1e-2, f"{D['D2_res']:.1e}")
    check("D3 half-dt ratio in [2.5,6]", 2.5 <= D["D3_ratio"] <= 6.0, f"{D['D3_ratio']:.2f}")
    check("D4 poisson residual <=1e-8", D["D4_poisson_res"] <= 1e-8, f"{D['D4_poisson_res']:.1e}")

    print(f"\n=== SELF-CHECK {'ALL GREEN' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} ===")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
