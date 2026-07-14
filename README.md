# adversarial_dns128 — self-contained 128³ DNS for adversarial ML

A self-contained package to **generate a 128³ incompressible-turbulence DNS dataset**,
**apply physically-legal adversarial perturbations to any frame**, and **step the solver
forward** to produce the true next frame / a full trajectory from the perturbed state.
Every produced field is checked against the **A-group + D-group DNS acceptance standard**.

Borrowed from a verified 256³ pseudo-spectral DNS solver (rotational-form incompressible
Navier–Stokes, RK3 + Lawson integrating factor for viscosity, 2/3-rule dealiasing,
Eswaran–Pope stochastic OU forcing, fp64 compute). This copy is trimmed to 128³ and is
**independent of any other project** — no external repo imports.

## Why produce locally instead of shipping data

The dataset is ~23 GB but **generation takes only minutes** (8 seeds × 120 frames ≈ 6 min
on one RTX 5090; faster on H200×4, one seed per GPU). Regenerating from this code is faster
and cleaner than transferring the data — and lets you change seeds/frames/Re freely, fully
reproducibly (same code + params ⇒ identical fields).

## Physics config (calibrated)

- **Grid**: 128³, fp64 compute, fp32 storage.
- **Re_λ ≈ 37**, k_maxη ≈ 1.60 → **Class I** (fully resolved; all statistics publishable).
  This is the highest Re that stays Class I on 128³ — Re_λ ≳ 50 pushes k_maxη < 1.5
  (Class II / fails A-group resolution). Set by `NU=0.006, k_f=4, sigma2=0.16` (in
  `generate_dataset.py`; k_f=4 keeps many forced modes so A10 isotropy pools cleanly).

## Requirements

`pip install -r requirements.txt` (torch ≥ 2.0, numpy, matplotlib, pyyaml). CUDA GPU with
fp64. On Windows set `KMP_DUPLICATE_LIB_OK=TRUE`. Run from the package root with the root on
`PYTHONPATH` (entry scripts `import solver`).

## Quick start

```bash
export PYTHONPATH=$PWD            # (Windows: set, or run from this dir)
export KMP_DUPLICATE_LIB_OK=TRUE  # Windows OpenMP guard; harmless elsewhere

# 0. one-shot self-check (solver + adversarial core + D-group, ~1 min)
python selfcheck.py

# 1. generate the dataset: 8 seeds × 120 frames -> data/dns128_relam37/
python generate_dataset.py --seeds 8 --frames 120

# 2. A + D acceptance on the frame-set
python eval/eval_ad.py data/dns128_relam37

# 3. adversarial single step: perturb a frame -> true next frame, check legality
python step_from_frame.py --frame data/dns128_relam37/seed00/frame060.pt --amp 0.10

# 4. adversarial PRODUCTION: perturb a frame -> run DNS forward from it (full trajectory)
python adversarial_produce.py --frame data/dns128_relam37/seed03/frame060.pt \
       --amp 0.10 --frames 60 --out data/adv_traj
python eval/eval_ad.py data/adv_traj

# 5. (optional) eyeball GIF of the flow
python scripts/make_gif.py data/dns128_relam37/seed00 data/flow.gif
```

## ⭐ The simple adversarial API — `advance(x)`

**One function.** You give a velocity field `x` (your original frame, **or** your frame with
an adversarial perturbation already added — anything), we return the **true DNS next frame**.
You do **not** need to make the perturbation physically valid: any `x` is first projected
onto the incompressible-DNS manifold (Leray + dealias), so whatever you did to it, we step a
legal DNS state and return its true evolution.

```python
from advance import advance

x_next = advance(x)          # x, x_next : (3, 128, 128, 128) numpy/torch, any real dtype
```

That's the whole interface. Options (all optional):

```python
x_next, info = advance(x, seed=0, return_info=True)
#   seed        : forcing realization (0..7 match dataset seeds; ou_seed = 1000+seed)
#   frame_dt    : sim-time to advance (default 0.30 = one dataset frame)
#   dt          : sub-step size (default: CFL-chosen — recommended)
#   return_info : also returns {div_residual, k_max_eta, K, legal}
#   info["legal"] is True when the returned frame is incompressible (div<1e-6) + Class I
```

If you'd rather **we** generate the legal adversarial example for you:

```python
from advance import advance, legal_perturb
x_adv  = legal_perturb(x, amp=0.10, seed=0)   # amp = relative rms perturbation budget
x_next = advance(x_adv)
```

`x` must be shape `(3, 128, 128, 128)` (channels = u, v, w). Run once from the package root
(`import advance` puts the solver on the path). First call builds the grid (~1 s); subsequent
calls reuse it.

## The adversarial internals (`step_from_frame.py`)

The adversarial-ML setting: an attacker takes one of our DNS frames and adds a
**physically-legal perturbation** (an adversarial example that stays on the
incompressible-DNS manifold), and we must return the **physically-correct next frame**.

- `load_frame_as_uhat(path)` — load any saved frame → solenoidal, dealiased spectral field.
- `legal_perturbation(u_hat, amp=...)` — add an adversary-chosen band-limited velocity
  field, then Leray-project + dealias the sum so the result is **exactly** a legal DNS state
  (∇·u = 0 to machine precision, no aliased modes). `amp` = relative rms perturbation budget.
- `step_one(u_hat)` — advance one true DNS step → next frame.
- `input_legality(u_hat)` — divergence residual, k_maxη, K, ε for any field.

"Legal" = incompressible (div residual ~1e-16) **and** resolved (Class I). A perturbation
of any adversary-chosen direction is accepted as long as the projected result is a legal
DNS field; the solver then returns the true evolution. `adversarial_produce.py` runs this
forward for many frames, producing a full DNS trajectory *started from* the adversarial
example — verified to stay Class I + incompressible the whole way.

## Acceptance (A + D)

`eval/eval_ad.py <case_dir>` reads `<case>/seed*/frame*.pt` and judges:

- **A1** resolution class — k_maxη ≥ 1.5 (Class I)
- **A2** resolved dissipation ≥ 99.5 %
- **A4** spectrum-tail monotonicity (0 upticks)
- **A10** component isotropy — cross ≤ 2 %, comp ≤ 5 % (pooled over all frames; high-variance,
  needs many frames × independent seeds — a single 60-frame trajectory can read ~4 %, the full
  8-seed × 120-frame set pools to ~0.13 %)
- **A13** derivative skewness (report; ~ −0.5)
- **D1** divergence residual ≤ 1e-6
- **D2** NS momentum residual ≤ 1e-2 (2h central-difference triplet, no forcing so the
  residual is pure time-truncation)
- **D3** half-dt convergence ratio ∈ [2.5, 6] (≈4 ⇒ O(h²), proving D2 is discretization
  truncation, not an equation error)
- **D4** velocity–pressure consistency ≤ 1e-8 (pressure Poisson residual)

The full 8-seed × 120-frame `dns128_relam37` set passes **A + D ALL PASS**
(A10 cross 0.128 %, D3 3.94, D4 7e-17).

## Layout

```
solver/                 trimmed 128³ pseudo-spectral DNS core (self-contained)
generate_dataset.py     produce N_SEEDS × N_FRAMES matured frames
step_from_frame.py      adversarial core: legal perturb + one-step + legality
adversarial_produce.py  DNS trajectory started from an adversarially-perturbed frame
eval/eval_ad.py         A-group + D-group acceptance referee
selfcheck.py            one-shot green/red self-check
scripts/make_gif.py     |omega| + u_x mid-plane GIF
```
Data (`data/`, *.pt, *.gif) is git-ignored — regenerate from code.
