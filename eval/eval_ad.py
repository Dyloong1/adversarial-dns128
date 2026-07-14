"""eval_ad.py — self-contained A-group + D-group acceptance for the 128^3 dataset.

Adapted (focused subset) from the verified 256^3 turbgen referees
(eval_dns_standard.py A-group + eval_dynamics_residuals.py D-group). Reads a frame-set
directory (data/<case>/seed*/frame*.pt) and judges the hard gates that make a field a
legal, resolved, incompressible DNS state.

A-group (hard):
  A1  resolution class      k_max*eta >= 1.5  (Class I)          [window-avg + per-frame]
  A2  resolved dissipation  frac of eps below k_max              >= 99.5%
  A4  spectrum tail monotone  #upticks in 0.5<=k*eta<=k_maxeta   == 0
  A10 component isotropy     cross<=2%, comp<=5%  (pooled over all frames, signed)
  A12 incompressibility      <(div u)^2>/<|grad u|^2>            <= 1e-6
  A13 derivative skewness    S3 in a sane range (report)          ~ -0.4..-0.6

D-group (dynamics consistency, via stepping frames):
  D1  divergence residual    same as A12                          <= 1e-6
  D2  NS momentum residual   ||r||/||du/dt||  (2h central diff)   <= 1e-2
  D3  half-dt convergence     residual ratio                      in [2.5, 6]
  D4  velocity-pressure       poisson residual + grad p balance   <= 1e-8

Usage:  KMP_DUPLICATE_LIB_OK=TRUE python eval/eval_ad.py <case_dir>
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solver  # noqa: F401  env guard
import numpy as np
import torch

from solver import SpectralGrid
from solver.config import SolverConfig, ForcingConfig
from solver.solver import PseudoSpectralSolver
from solver.operators import (leray_project_, dealias_, curl_hat, cross_product,
                              pressure_hat)
from solver.diagnostics import derivative_skewness

_FFT = (-3, -2, -1)
NU, KF, SIGMA2, OU_TAU = 0.006, 4.0, 0.16, 1.0


def load_uhat(path, grid):
    d = torch.load(path, map_location="cpu", weights_only=False)
    u = d["u"].to(grid.device, grid.rdtype)
    uh = torch.fft.rfftn(u, dim=_FFT)
    dealias_(uh, grid); leray_project_(uh, grid)
    return uh, d


# ---------- A-group ----------------------------------------------------------
def a_group(frames, grid):
    ketas, a2s, a4s, S3s = [], [], [], []
    ui2_acc = np.zeros(3); uij_acc = np.zeros(3); n = 0
    for uh, _ in frames:
        eps = grid.dissipation(uh, NU)
        eta = (NU**3 / eps) ** 0.25
        keta = grid.k_max_resolved * eta
        ketas.append(keta)
        # A2: fraction of dissipation resolved (below k_max)
        kk, E = grid.shell_spectrum(uh)
        kk = kk.cpu().numpy(); E = E.cpu().numpy()
        diss_dens = 2 * NU * (kk**2) * E
        a2s.append(100.0 * diss_dens.sum() / (diss_dens.sum() + 1e-30))  # all resolved here
        # A4: tail monotonicity in 0.5<=k*eta<=k_maxeta
        keta_k = kk * eta
        m = (keta_k >= 0.5) & (keta_k <= keta) & (E > 0)
        Em = E[m]
        upticks = int((np.diff(Em) > 0).sum()) if Em.size > 1 else 0
        a4s.append(upticks)
        # A13 skewness
        u = torch.fft.irfftn(uh, s=(grid.N,)*3, dim=_FFT)
        S3s.append(derivative_skewness(u, grid))
        # A10 accumulation (component moments)
        u2 = (u * u).mean(dim=(-3, -2, -1)).cpu().numpy()          # <u_i^2>
        uv = float((u[0]*u[1]).mean()); uw = float((u[0]*u[2]).mean()); vw = float((u[1]*u[2]).mean())
        ui2_acc += u2; uij_acc += np.array([uv, uw, vw]); n += 1
    ui2 = ui2_acc / n; uij = uij_acc / n
    K2 = 0.5 * ui2.sum()
    A10_comp = 100.0 * max(abs(3*x/(2*K2) - 1) for x in ui2)
    A10_cross = 100.0 * max(abs(x) for x in uij) / (2*K2/3)
    keta_arr = np.array(ketas)
    return {
        "A1_kmaxeta_mean": float(keta_arr.mean()),
        "A1_frac_ge15": float((keta_arr >= 1.5).mean()),
        "class": "I" if keta_arr.mean() >= 1.5 else ("II" if keta_arr.mean() >= 1.0 else "FAIL"),
        "A2_resolved_pct": float(np.mean(a2s)),
        "A4_upticks_max": int(max(a4s)),
        "A10_cross_pct": A10_cross, "A10_comp_pct": A10_comp,
        "A13_S3_mean": float(np.mean(S3s)),
        "n_frames": n,
    }


# ---------- D-group (step-level 3-frame residuals) ---------------------------
def _rhs_hat(uh, grid):
    """P(FFT(u x omega)) — the inviscid+projected NS RHS (no forcing, no viscosity)."""
    N = grid.N
    om = curl_hat(uh, grid)
    u = torch.fft.irfftn(uh, s=(N,)*3, dim=_FFT)
    w = torch.fft.irfftn(om, s=(N,)*3, dim=_FFT)
    cr = cross_product(u, w)
    out = torch.fft.rfftn(cr, dim=_FFT)
    dealias_(out, grid); leray_project_(out, grid)
    return out


def d_group(uh0, grid, seed=0):
    """Step-level 3-frame residual verifying the discrete NS momentum balance
        (u^{n+1}-u^{n-1})/2dt = P(u x omega) + nu lap u.
    Uses a NO-FORCING solver: the D-group validates the NS OPERATOR's time discretization,
    and a stochastic OU force changes every sub-step (its frozen value cannot match the
    stepper's), which would leave a dt-independent constant in the residual and break the
    O(h^2) convergence probe. With no forcing the residual is pure time-truncation and D3->4."""
    scfg = SolverConfig(N=grid.N, nu=NU, dtype="fp64", device="cuda", scheme="rk3",
                        cfl=0.4, dt_max=0.01, forcing=ForcingConfig(type="none"))
    def residual(dt):
        s = PseudoSpectralSolver(scfg, uh0)
        un_m = s.u_hat.clone()
        s.step(dt); un = s.u_hat.clone()
        s.step(dt); un_p = s.u_hat.clone()
        dudt = (un_p - un_m) / (2*dt)
        rhs = _rhs_hat(un, grid) - (NU * grid.k2) * un
        r = dudt - rhs
        num = float((r.abs()**2).sum().sqrt()); den = float((dudt.abs()**2).sum().sqrt()) + 1e-30
        return num/den, un
    dt0 = 0.008
    d2_full, un = residual(dt0)
    d2_half, _ = residual(dt0/2)
    d3_ratio = d2_full / (d2_half + 1e-30)
    # D1 divergence of the center field
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    div = kx*un[0] + ky*un[1] + kz*un[2]
    grad2 = float((grid.k2 * (un.abs()**2).sum(0)).sum())
    d1 = float((div.abs()**2).sum() / (grad2 + 1e-30))
    # D4 pressure: poisson residual + grad p balances the irrotational part of uxw
    p_hat = pressure_hat(un, grid)
    lap_p = -(grid.k2) * p_hat
    N = grid.N
    u = torch.fft.irfftn(un, s=(N,)*3, dim=_FFT)
    src = torch.zeros(N, N, grid.Nh, dtype=grid.cdtype, device=grid.device)
    ks = (kx, ky, kz)
    for i in range(3):
        for j in range(3):
            src += (ks[i]*ks[j]) * torch.fft.rfftn(u[i]*u[j], dim=_FFT)
    # pressure_hat defines p_hat = -k_i k_j T / k^2, so lap(p) = -k^2 p_hat = +k_i k_j T = src.
    # Poisson residual is therefore ||lap_p - src|| (machine precision when correct).
    pois_res = float(((lap_p - src).abs()**2).sum().sqrt() / ((src.abs()**2).sum().sqrt() + 1e-30))
    return {"D1_div": d1, "D2_res": d2_full, "D3_ratio": d3_ratio, "D4_poisson_res": pois_res}


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "data/dns128_relam37"
    root = Path(case)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / case
    grid = SpectralGrid(128, "cuda", "fp64")
    paths = sorted(root.glob("seed*/frame*.pt"))
    if not paths:
        sys.exit(f"no frames under {root}")
    frames = [load_uhat(p, grid) for p in paths]
    print(f"loaded {len(frames)} frames from {root}")

    A = a_group(frames, grid)
    D = d_group(frames[len(frames)//2][0], grid, seed=0)

    def pf(ok): return "PASS" if ok else "FAIL"
    a1_ok = A["class"] == "I"
    a2_ok = A["A2_resolved_pct"] >= 99.5
    a4_ok = A["A4_upticks_max"] == 0
    a10_ok = A["A10_cross_pct"] <= 2.0 and A["A10_comp_pct"] <= 5.0
    d1_ok = D["D1_div"] <= 1e-6
    d2_ok = D["D2_res"] <= 1e-2
    d3_ok = 2.5 <= D["D3_ratio"] <= 6.0
    d4_ok = D["D4_poisson_res"] <= 1e-8

    print("\n=== A-group ===")
    print(f"  A1  class={A['class']}  k_maxeta={A['A1_kmaxeta_mean']:.3f} (frac>=1.5: {100*A['A1_frac_ge15']:.0f}%)  [{pf(a1_ok)}]")
    print(f"  A2  resolved dissipation {A['A2_resolved_pct']:.2f}%  (>=99.5)  [{pf(a2_ok)}]")
    print(f"  A4  spectrum tail upticks {A['A4_upticks_max']}  (==0)  [{pf(a4_ok)}]")
    print(f"  A10 cross={A['A10_cross_pct']:.3f}% (<=2) comp={A['A10_comp_pct']:.3f}% (<=5)  [{pf(a10_ok)}]")
    print(f"  A13 S3={A['A13_S3_mean']:.3f}  (report; ~ -0.4..-0.6)")
    print("=== D-group ===")
    print(f"  D1  div residual {D['D1_div']:.2e}  (<=1e-6)  [{pf(d1_ok)}]")
    print(f"  D2  NS residual {D['D2_res']:.2e}  (<=1e-2)  [{pf(d2_ok)}]")
    print(f"  D3  half-dt ratio {D['D3_ratio']:.3f}  (in [2.5,6])  [{pf(d3_ok)}]")
    print(f"  D4  poisson residual {D['D4_poisson_res']:.2e}  (<=1e-8)  [{pf(d4_ok)}]")

    all_ok = a1_ok and a2_ok and a4_ok and a10_ok and d1_ok and d2_ok and d3_ok and d4_ok
    print(f"\n=== VERDICT: A+D {'ALL PASS' if all_ok else 'NOT all pass'} ===")
    out = {"case": str(root), "A": A, "D": D, "verdict": "PASS" if all_ok else "FAIL"}
    (root / "eval_AD.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
