"""adversarial_produce.py — produce a DNS trajectory STARTING FROM an adversarially
perturbed frame.

This is the "adversarial DNS production" use case: take one of our legal DNS frames,
apply a physically-legal adversarial perturbation, and then run the SAME DNS solver
forward from that perturbed state to produce a full trajectory (many frames). Every
produced frame is a true DNS evolution of the perturbed input and must remain legal
(incompressible + Class I) and pass A+D. This proves the adversarial example is not a
one-off single-step curiosity but a valid initial condition for real DNS production.

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python adversarial_produce.py --frame <frame.pt> \
      --amp 0.10 --frames 60 --out data/adv_traj
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import solver  # noqa env guard
import torch

from solver import SpectralGrid, PseudoSpectralSolver
from solver.config import SolverConfig, ForcingConfig
from step_from_frame import legal_perturbation, load_frame_as_uhat, input_legality

N, NU, KF, SIGMA2, OU_TAU = 128, 0.006, 4.0, 0.16, 1.0
FRAME_DT = 0.30
HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="seed frame*.pt to perturb")
    ap.add_argument("--amp", type=float, default=0.10, help="adversarial perturbation strength")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="forcing seed for the continued run")
    ap.add_argument("--out", default=str(HERE / "data" / "adv_traj"))
    args = ap.parse_args()

    grid = SpectralGrid(N, "cuda", "fp64")
    # 1. load a legal frame, apply a legal adversarial perturbation
    u0 = load_frame_as_uhat(args.frame, grid)
    up = legal_perturbation(u0, grid, amp=args.amp, seed=args.seed + 777)
    L0 = input_legality(u0, grid); Lp = input_legality(up, grid)
    d_rms = math.sqrt(2*grid.kinetic_energy(up - u0)/3.0)
    base_rms = math.sqrt(2*grid.kinetic_energy(u0)/3.0)
    print(f"base frame:      div={L0['div_residual']:.1e} k_maxeta={L0['k_max_eta']:.3f}")
    print(f"adversarial in:  div={Lp['div_residual']:.1e} k_maxeta={Lp['k_max_eta']:.3f} "
          f"|delta|/|u|={d_rms/base_rms:.3f}")

    # 2. run the SAME DNS solver forward from the perturbed state
    scfg = SolverConfig(N=N, nu=NU, dtype="fp64", device="cuda", scheme="rk3",
                        cfl=0.4, dt_max=0.01,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=KF, ou_tau=OU_TAU,
                                              ou_sigma2=SIGMA2, ou_seed=1000 + args.seed))
    s = PseudoSpectralSolver(scfg, up)

    out_dir = Path(args.out) / "seed00"     # single trajectory; eval expects seed*/frame*
    out_dir.mkdir(parents=True, exist_ok=True)
    ketas = []
    worst_div = 0.0
    for fi in range(args.frames):
        t_target = s.t + FRAME_DT
        while s.t < t_target:
            s.step(s.suggest_dt(t_target))
        L = input_legality(s.u_hat, grid)
        ketas.append(L["k_max_eta"]); worst_div = max(worst_div, L["div_residual"])
        u = s.velocity_physical().to(torch.float32).cpu()
        torch.save({"u": u, "t": float(s.t), "k_max_eta": L["k_max_eta"], "seed": 0,
                    "nu": NU, "N": N}, out_dir / f"frame{fi:03d}.pt")
        if (fi + 1) % 20 == 0:
            print(f"  frame{fi:03d} t={s.t:.2f} k_maxeta={L['k_max_eta']:.3f} "
                  f"div={L['div_residual']:.1e} K={L['K']:.3f}", flush=True)

    ketas = torch.tensor(ketas)
    all_classI = bool((ketas >= 1.5).all())
    print(f"\n=== adversarial trajectory: {args.frames} frames ===")
    print(f"  k_maxeta: min={float(ketas.min()):.3f} mean={float(ketas.mean()):.3f} "
          f"(all >=1.5: {all_classI})")
    print(f"  worst divergence residual over trajectory: {worst_div:.1e} (<=1e-6: {worst_div<=1e-6})")
    print(f"  => adversarially-started DNS production {'VALID' if all_classI and worst_div<=1e-6 else 'INVALID'}")
    (Path(args.out) / "manifest.json").write_text(json.dumps({
        "N": N, "nu": NU, "k_f": KF, "started_from": str(args.frame), "adv_amp": args.amp,
        "frames": args.frames, "all_class_I": all_classI, "worst_div": worst_div,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
