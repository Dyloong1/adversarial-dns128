"""test_adversarial.py — construct diverse perturbations and verify BOTH service paths:

  advance_raw       : must STEP legal adversarial inputs, and REFUSE/flag illegal ones
                      (compressible / aliased / under-resolved) — never silently "fix".
  advance_projected : must accept anything and return a legal next frame.
  input_legality    : must correctly classify each perturbation type.

This is the evidence that the no-projection (attacker-decides-legality) path behaves
honestly: it does not secretly purify the attacker's field.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python tests/test_adversarial.py
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solver  # noqa env guard
import numpy as np
import torch

from advance import (advance_raw, advance_projected, input_legality, legal_perturb,
                     project_to_manifold, _grid, N, _FFT)
from solver import PseudoSpectralSolver
from solver.config import SolverConfig, ForcingConfig
from solver.initial_conditions import random_solenoidal
from solver.operators import leray_project_, dealias_

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok: FAIL.append(name)


def matured_frame():
    grid = _grid()
    s = PseudoSpectralSolver(
        SolverConfig(N=N, nu=0.006, dtype="fp64",
                     device="cuda" if torch.cuda.is_available() else "cpu",
                     scheme="rk3", cfl=0.4, dt_max=0.01,
                     forcing=ForcingConfig(type="stochastic_ou", k_f=4.0, ou_tau=1.0,
                                           ou_sigma2=0.16, ou_seed=1000)),
        random_solenoidal(grid, seed=3, k_p=4.0, u_rms=0.7))
    while s.t < 15.0:
        s.step(s.suggest_dt(15.0))
    return torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=_FFT).cpu().numpy(), grid


def main():
    print("=== adversarial perturbation test suite ===")
    x, grid = matured_frame()
    L = input_legality(x)
    check("base matured frame is legal", L["legal"], f"div={L['div_residual']:.1e} keta={L['k_max_eta']:.3f}")

    # --- perturbation type 1: LEGAL adversarial (solenoidal, band-limited) ---
    for amp in (0.02, 0.10, 0.25):
        xa = legal_perturb(x, amp=amp, seed=7)
        La = input_legality(xa)
        check(f"legal_perturb amp={amp} classified legal", La["legal"],
              f"div={La['div_residual']:.1e} comp_frac={La['compressible_fraction']:.1e}")
        # raw must STEP it without complaint and return a legal next frame
        try:
            xn, info = advance_raw(xa, return_info=True)
            ok = info["next"]["legal"]
        except Exception as e:
            ok = False; info = {"err": str(e)}
        check(f"advance_raw steps legal amp={amp} -> legal next", ok)

    # --- perturbation type 2: COMPRESSIBLE (adds a gradient/dilatational field) ---
    # x_comp = x + grad(phi): pure compressible perturbation, NOT solenoidal
    phi = torch.randn(N, N, N, dtype=torch.float64, device=grid.device)
    phi_hat = torch.fft.rfftn(phi, dim=_FFT)
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    gx = torch.fft.irfftn(1j*kx*phi_hat, s=(N,)*3, dim=_FFT)
    gy = torch.fft.irfftn(1j*ky*phi_hat, s=(N,)*3, dim=_FFT)
    gz = torch.fft.irfftn(1j*kz*phi_hat, s=(N,)*3, dim=_FFT)
    grad = torch.stack([gx, gy, gz]).cpu().numpy()
    grad *= 0.1 * float(np.sqrt((x**2).mean())) / float(np.sqrt((grad**2).mean()))
    x_comp = x + grad.astype(np.float32)
    Lc = input_legality(x_comp)
    check("compressible perturbation flagged ILLEGAL", not Lc["legal"],
          f"comp_frac={Lc['compressible_fraction']:.2e} div={Lc['div_residual']:.1e}")
    # raw must REFUSE (raise) by default — NOT silently project
    refused = False
    try:
        advance_raw(x_comp)
    except ValueError:
        refused = True
    check("advance_raw REFUSES compressible input (no silent fix)", refused)
    # projected path must accept it and return a legal frame
    xn = advance_projected(x_comp)
    check("advance_projected accepts compressible -> legal next", input_legality(xn)["legal"])
    # and projecting the compressible input removes the dilatational part
    xp = project_to_manifold(x_comp)
    check("project_to_manifold strips compressible part", input_legality(xp)["legal"],
          f"comp_frac {Lc['compressible_fraction']:.1e} -> {input_legality(xp)['compressible_fraction']:.1e}")

    # --- perturbation type 3: HIGH-FREQUENCY / aliased (energy above the 2/3 band) ---
    uh = torch.fft.rfftn(torch.as_tensor(x).to(grid.device, grid.rdtype), dim=_FFT)
    hf = torch.randn_like(uh) * (~grid.dealias_mask)          # energy only above dealias cut
    hf = hf / (hf.abs().pow(2).sum().sqrt() + 1e-30) * uh.abs().pow(2).sum().sqrt() * 0.1
    x_hf = torch.fft.irfftn(uh + hf, s=(N,)*3, dim=_FFT).to(torch.float32).cpu().numpy()
    Lh = input_legality(x_hf)
    check("aliased (high-freq) perturbation flagged ILLEGAL", not Lh["legal"],
          f"aliased_frac={Lh['aliased_fraction']:.2e}")
    refused_hf = False
    try:
        advance_raw(x_hf)
    except ValueError:
        refused_hf = True
    check("advance_raw REFUSES aliased input", refused_hf)

    # --- END-TO-END on a REAL dataset frame loaded from disk (the actual user flow) ---
    import glob
    disk = sorted(glob.glob(str(Path(__file__).resolve().parents[1] /
                                "data" / "dns128_relam37" / "seed*" / "frame*.pt")))
    if disk:
        import numpy as _np
        d = torch.load(disk[len(disk)//2], map_location="cpu", weights_only=False)
        xr = d["u"].numpy()                          # fp32 frame straight off disk
        check("real disk frame loads legal", input_legality(xr)["legal"],
              f"stored k_maxeta={d.get('k_max_eta', float('nan')):.3f}")
        # modify it (legal adversarial) -> advance_raw -> next frame
        xr_adv = legal_perturb(xr, amp=0.10, seed=42)
        xn, info = advance_raw(xr_adv, return_info=True)
        check("disk frame + legal perturb -> advance_raw -> legal next", info["next"]["legal"],
              f"div={info['next']['div_residual']:.1e} keta={info['next']['k_max_eta']:.3f}")
        moved = float(np.abs(xn - xr_adv).max())
        check("next frame is a real evolution (not identity)", moved > 1e-3, f"max|dx|={moved:.2f}")
        # illegal modification of a real frame -> raw refuses, projected accepts
        xr_bad = xr + 0.05 * np.random.RandomState(0).randn(*xr.shape).astype("float32")
        refused_r = False
        try:
            advance_raw(xr_bad)
        except ValueError:
            refused_r = True
        check("real frame + illegal modification -> raw REFUSES", refused_r)
        check("real frame + illegal modification -> projected returns legal next",
              input_legality(advance_projected(xr_bad))["legal"])
    else:
        print("  [skip] no data/dns128_relam37 on disk — run generate_dataset.py to enable "
              "the real-frame end-to-end test")

    # --- perturbation type 4: on_illegal='warn' proceeds; 'ignore' proceeds silently ---
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        advance_raw(x_comp, on_illegal="warn")
        check("on_illegal='warn' proceeds with a warning", len(w) >= 1)

    print(f"\n=== ADVERSARIAL TEST {'ALL GREEN' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} ===")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
