"""generate_dataset.py — self-contained 128^3 fp64 DNS dataset generator.

Borrowed from the verified 256^3 turbgen solver (pseudo-spectral, rotational-form
incompressible NS, RK3 + Lawson integrating factor, 2/3 dealiasing, Eswaran-Pope OU
stochastic forcing). Produces a matured (statistically stationary) frame-set for
N_SEEDS independent seeds at Re_lambda~37 (k_maxeta~1.56, Class I on 128^3).

Each frame is a solenoidal, dealiased velocity field saved as fp32 (u [3,128,128,128])
plus t and instantaneous k_max*eta. Frames start AFTER spin-up so they are on the
stationary attractor — any frame is a valid DNS state that step_from_frame.py can
advance and eval/ can accept.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python generate_dataset.py [--seeds 8] [--frames 20]
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import solver  # noqa: F401  (env guard: must precede torch import path)
import torch

from solver import SpectralGrid, PseudoSpectralSolver
from solver.config import SolverConfig, ForcingConfig

# ---- calibrated 128^3 config (Re_lambda~37, k_maxeta~1.56, Class I) ------------
N        = 128
NU       = 0.006
KF       = 4.0
SIGMA2   = 0.16
OU_TAU   = 1.0
OU_SEED0 = 1000       # per-seed ou_seed = OU_SEED0 + seed (independent forcing)
U_RMS0   = 0.7
K_P      = 4.0
T_SPINUP = 15.0       # reach stationarity before sampling
FRAME_DT = 0.30       # sim-time between exported frames (~a few tau_eta)
HERE     = Path(__file__).parent


def make_solver(seed: int) -> PseudoSpectralSolver:
    grid = SpectralGrid(N, "cuda", "fp64")
    scfg = SolverConfig(
        N=N, nu=NU, dtype="fp64", device="cuda", scheme="rk3", cfl=0.4, dt_max=0.01,
        forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                              ou_sigma2=SIGMA2, ou_seed=OU_SEED0 + seed),
    )
    from solver.initial_conditions import random_solenoidal
    u_hat0 = random_solenoidal(grid, seed=seed, k_p=K_P, u_rms=U_RMS0)
    return PseudoSpectralSolver(scfg, u_hat0)


def kmaxeta(s: PseudoSpectralSolver) -> float:
    eps = s.grid.dissipation(s.u_hat, NU)
    eta = (NU**3 / eps) ** 0.25
    return s.grid.k_max_resolved * eta


def run_seed(seed: int, n_frames: int, out_dir: Path) -> dict:
    s = make_solver(seed)
    # spin-up to stationarity
    while s.t < T_SPINUP:
        s.step(s.suggest_dt(T_SPINUP))
    seed_dir = out_dir / f"seed{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    frames_meta = []
    for fi in range(n_frames):
        t_target = s.t + FRAME_DT
        while s.t < t_target:
            s.step(s.suggest_dt(t_target))
        u = s.velocity_physical().to(torch.float32).cpu()   # [3,N,N,N]
        ke = kmaxeta(s)
        torch.save({"u": u, "t": float(s.t), "k_max_eta": float(ke), "seed": seed,
                    "nu": NU, "N": N}, seed_dir / f"frame{fi:03d}.pt")
        frames_meta.append({"frame": fi, "t": float(s.t), "k_max_eta": float(ke)})
        print(f"  seed{seed:02d} frame{fi:03d} t={s.t:.2f} k_maxeta={ke:.3f} "
              f"K={s.scalars()['K']:.3f}", flush=True)
    return {"seed": seed, "ou_seed": OU_SEED0 + seed, "n_frames": n_frames,
            "frames": frames_meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", default=str(HERE / "data" / "dns128_relam37"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"N": N, "nu": NU, "k_f": KF, "sigma2": SIGMA2, "ou_tau": OU_TAU,
                "target": "Re_lambda~37, k_maxeta~1.56 Class I", "frame_dt": FRAME_DT,
                "t_spinup": T_SPINUP, "seeds": []}
    for seed in range(args.seeds):
        print(f"=== seed {seed} (spinup {T_SPINUP} + {args.frames} frames) ===", flush=True)
        manifest["seeds"].append(run_seed(seed, args.frames, out_dir))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDONE: {args.seeds} seeds x {args.frames} frames -> {out_dir}")


if __name__ == "__main__":
    main()
