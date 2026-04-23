"""
Risk-Bounded Lane-Change Controller  ── FINAL CORRECTED VERSION
================================================================
Based on: "Risk-Bounded Control Using Stochastic Barrier Functions"
          Yaghoubi et al., IEEE Control Systems Letters, 2021

Corrections applied
--------------------
FIX 1 – sigma_acc / sigma_target: use γ (barrier gain) not road slope.
         σ_acc    = (1/2)·γ²·cᵢ²·B
         σ_target = (B/2)[γ²(dh·go)² − γ·dot(d2h_diag, go²)]
         Negative σ is clamped to 0 (self-stabilising → conservative bound).

FIX 2 – vc vs speed: vc_i (nominal drift) estimated as a smoothed speed,
         not the instantaneous noisy speed, to keep LfH_xo consistent with
         the paper's SDE model (Eq. 24).

FIX 3 – Noise injection: removed the erroneous ×5 amplification.
         Wiener increment is now N(0, dt) scaled only by go = cᵢ·[1, γ].

FIX 4 – p_bar raised from 1e-8 → 0.1 (paper's experimental value, Sec. VI).
         1e-8 made b_max ≈ 0 at all times, causing near-constant infeasibility.

FIX 5 – Multi-agent SCBF: all vehicles within proximity_radius (not just
         one target) contribute SCBF constraints, each with their own b slot.
         QP decision vector generalised to [a, β, δ_v, δ_y, b_0, …, b_{M-1}].

FIX 6 – Risk logging: worst-case upper bound across all active agents.

State / control convention
--------------------------
  xr = [X, Y, v, ψ]      (ego, deterministic; kinematic bicycle)
  xo = [Xo, Yo]           (each obstacle, stochastic)
  u  = [a, β]             (acceleration, slip-angle ≈ steering proxy)
  QP z = [a, β, δ_v, δ_y, b_0, …, b_{M-1}]
"""

import gymnasium as gym
import highway_env           # noqa: F401  (registers envs)
import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import pygame

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
COLOR_DEFAULT = (160, 160, 160)
COLOR_TARGET  = (255,   0,   0)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
params = {
    # Environment geometry
    "v_d":          25.0,
    "lane_width":    4.0,
    "N_lanes":       3,
    "l_r":           5.0,       # bicycle rear-axle offset
    "y_min":         0.0,
    "y_max":        14.0,

    # CLF decay rates
    "clf_rate_v":    0.70,
    "clf_rate_y":    0.15,

    # Deterministic CBF gain (lane-boundary only — no stochastic noise on walls)
    "cbf_k":         2.0,
    "cbf_k_acc":     3.0,
    "cbf_k_lane":   10.0,

    # Input limits
    "a_max":         2.5,
    "a_min":        -4.0,
    "beta_max":      0.5,
    "beta_min":     -0.5,

    # QP cost weights
    "w_a":           0.10,
    "w_beta":     2000.0,
    "w_delta_v":  5000.0,
    "w_delta_y":     1.0,
    "w_b":         100.0,   # penalise b — drives b small → tighter risk

    # Physics / timing
    "T":             0.1,   # finite-time planning horizon [s] for risk bound
    "v0":           15.0,   # reference speed for ACC formula
    "cd":            0.35,
    "g":             9.81,

    # Ellipse half-axes for target-vehicle SCBF
    "Bx":            7.0,
    "By":            4.0,

    # Barrier exponential gain  B(x) = exp(−γ h(x))
    "gamma":         5.0,

    # Obstacle noise intensity (paper: cᵢ in Eq. 24)
    "c_i":           0.1,

    # SCBF parameters (Theorem 2, condition 14)
    "a_scbf":        1.0,   # fixed a > 0; paper fixes a and solves for b
    "p_bar":         0.1,   # desired probability-of-failure upper bound (paper Sec. VI uses 0.1)

    # Multi-agent SCBF proximity threshold (paper: only constrain agents within dist 3)
    "proximity_radius": 15.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Lane geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def lane_center_from_index(k, p=params):
    return k * p["lane_width"] + p["lane_width"] / 2 - 2

def lane_index_from_y(y, p=params):
    idx = int(np.round(y / p["lane_width"]))
    return int(np.clip(idx, 0, p["N_lanes"] - 1))

# ─────────────────────────────────────────────────────────────────────────────
# Ego (deterministic) dynamics  xr = [X, Y, v, ψ]   u = [a, β]
# Kinematic bicycle (derivation doc §1):
#   ẋr = f(xr) + g(xr) u
# ─────────────────────────────────────────────────────────────────────────────
def f_vec(xr):
    _, _, v, psi = xr
    return np.array([v * np.cos(psi),
                     v * np.sin(psi),
                     0.0,
                     0.0])

def g_mat(xr):
    _, _, v, psi = xr
    return np.array([
        [0.0, -v * np.sin(psi)],
        [0.0,  v * np.cos(psi)],
        [1.0,  0.0            ],
        [0.0,  v / params["l_r"]],
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Nominal-velocity estimator for stochastic agents
# Paper Eq. (24): fo = [vc, γ·vc]ᵀ where vc is the *constant nominal* speed.
# Using the raw instantaneous speed conflates deterministic drift with noise.
# We maintain a per-vehicle exponential moving average as a proxy for vc.
# ─────────────────────────────────────────────────────────────────────────────
_vc_estimates: dict = {}   # id(veh) → smoothed speed estimate

def get_vc_estimate(veh, alpha: float = 0.05) -> float:
    """Exponential moving average of vehicle speed  →  proxy for paper's vc,i."""
    key = id(veh)
    current = float(veh.speed)
    if key not in _vc_estimates:
        _vc_estimates[key] = current
    else:
        _vc_estimates[key] = (1.0 - alpha) * _vc_estimates[key] + alpha * current
    return _vc_estimates[key]



def get_slope(veh):
    """Lateral/longitudinal velocity ratio (road slope γ_i in the paper)."""
    vx = veh.speed * np.cos(veh.heading)
    vy = veh.speed * np.sin(veh.heading)
    return 0.0 if abs(vx) < 1e-6 else vy / vx

def f_sto(veh):
    """Nominal drift using smoothed speed estimate (not raw instantaneous speed).
    FIX 2: vc,i in paper Eq.(24) is a constant nominal speed, not the noisy
    instantaneous reading. Using get_vc_estimate() avoids conflating drift with noise."""
    slope = get_slope(veh)
    vc    = get_vc_estimate(veh)
    return np.array([vc, slope * vc])

def g_sto(veh, c_i=None):
    """go column vector — shape (2,).  Paper Eq. (24): go = cᵢ·[1, γ_slope]ᵀ"""
    if c_i is None:
        c_i = params["c_i"]
    return c_i * np.array([1.0, get_slope(veh)])

# ─────────────────────────────────────────────────────────────────────────────
# Barrier function  B(x) = exp(−γ h(x))
# ─────────────────────────────────────────────────────────────────────────────
def barrier_value(h, gamma=None):
    if gamma is None:
        gamma = params["gamma"]
    return float(np.exp(-gamma * h))

# ─────────────────────────────────────────────────────────────────────────────
# Stochastic correction  σ = (1/2) tr( goᵀ · ∂²B/∂xo² · go )
#
# For B = exp(−γ h):
#   ∂B/∂xo   = −γ B (∂h/∂xo)
#   ∂²B/∂xo² = γ² B (∂h/∂xo)(∂h/∂xo)ᵀ − γ B (∂²h/∂xo²)
#
# Therefore:
#   σ = (1/2) goᵀ [γ²B (∂h/∂xo)(∂h/∂xo)ᵀ − γB ∂²h/∂xo²] go
#     = (B/2) [γ²(∂h/∂xo · go)² − γ goᵀ diag(∂²h/∂xo²) go]
#
# ─── ACC barrier  h = Xo − X − Tv − (v0−v)²/(2cd g) − 5 ─────────────────
#   ∂h/∂xo = [1, 0],  ∂²h/∂xo² = 0
#   σ_acc  = (B/2) · γ² · (go[0])² = (B/2) · γ² · cᵢ²
#   (derivation doc §6.4)
#
# ─── Target-ellipse barrier  h = (xi−X)²/Bx² + (yi−Y)²/By² − 1 ──────────
#   ∂h/∂xo = [+2(xi−X)/Bx², +2(yi−Y)/By²]  (xo IS the obstacle position)
#   ∂²h/∂xo² = diag(2/Bx², 2/By²)
#   σ_target = (B/2)[γ²(dh·go)² − γ dot(d2h_diag, go²)]
#   (derivation doc §8.2)
# ─────────────────────────────────────────────────────────────────────────────

def sigma_acc(h_val, veh, p=params):
    """
    FIX (Bug 1): use γ² (barrier gain), NOT slope².
    σ_acc = (1/2) · γ² · cᵢ² · B
    """
    B     = barrier_value(h_val, p["gamma"])
    gamma = p["gamma"]
    c_i   = p["c_i"]
    # go[0] = cᵢ · 1  (X component);  ∂h/∂xo=[1,0]  ⟹  (dh·go)² = cᵢ²
    return 0.5 * gamma**2 * c_i**2 * B


def sigma_target(h_val, xi, yi, xr, veh, p=params):
    """
    σ_target = (B/2)[γ²(dh·go)² − γ·dot(d2h_diag, go²)]
    Negative σ is clamped to 0: a negative Itô term means the stochastic
    dynamics are self-stabilising, so the deterministic bound is already
    conservative. Clamping avoids artificially tightening the QP constraint.
    """
    X, Y  = xr[0], xr[1]
    B     = barrier_value(h_val, p["gamma"])
    go    = g_sto(veh, p["c_i"])   # shape (2,)
    gamma = p["gamma"]
    Bx, By = p["Bx"], p["By"]
    dx, dy  = xi - X, yi - Y

    dh_dxo   = np.array([2.0 * dx / Bx**2, 2.0 * dy / By**2])
    d2h_diag = np.array([2.0 / Bx**2,      2.0 / By**2])

    term1 = gamma**2 * (dh_dxo @ go)**2
    term2 = gamma    * np.dot(d2h_diag, go**2)
    # FIX: clamp to 0 — negative σ means self-stabilising noise; use 0 (conservative)
    return max(0.0, 0.5 * B * (term1 - term2))

# ─────────────────────────────────────────────────────────────────────────────
# Risk-bound on b  (Theorem 2, condition 14)
#
#   b ≤ min( a,  −(1/T) ln((1 − p̄)/(1 − B0)) )
#
# Clamped to [0, a_scbf] to prevent infeasibility when the vehicle is already
# in or near the unsafe set (B0 large → log argument < 1 → b_max < 0).
# In that case the QP is best-effort; a backup controller should take over.
# ─────────────────────────────────────────────────────────────────────────────
def compute_b_max(B0, p=params):
    """
    FIX (Bug 5): clamp b_max to [0, a_scbf].
    """
    a    = p["a_scbf"]
    T    = p["T"]
    pbar = p["p_bar"]
    ratio = (1.0 - pbar) / max(1.0 - B0, 1e-9)
    if ratio <= 0.0:
        return float(a)                          # already violated; keep feasible
    b_cond = -(1.0 / T) * np.log(ratio)
    return float(np.clip(min(a, b_cond), 0.0, a))   # FIX: clamp ≥ 0

# ─────────────────────────────────────────────────────────────────────────────
# CLFs
# ─────────────────────────────────────────────────────────────────────────────
def clf_v(xr):
    return (xr[2] - params["v_d"]) ** 2

def clf_y(xr, y_t):
    return (xr[1] - y_t) ** 2

def lie_clf_v(xr):
    dV = np.array([0.0, 0.0, 2.0 * (xr[2] - params["v_d"]), 0.0])
    return clf_v(xr), dV @ f_vec(xr), dV @ g_mat(xr)

def lie_clf_y(xr, y_t):
    dV = np.array([0.0, 2.0 * (xr[1] - y_t), 0.0, 0.0])
    return clf_y(xr, y_t), dV @ f_vec(xr), dV @ g_mat(xr)

# ─────────────────────────────────────────────────────────────────────────────
# Lane-boundary CBF  (deterministic; no stochastic correction needed)
# ─────────────────────────────────────────────────────────────────────────────
def cbf_boundary(xr):
    y = xr[1]
    return -np.log(
        np.exp(-(y - params["y_min"])) +
        np.exp(-(params["y_max"] - y))
    )

def lie_cbf_boundary(xr):
    y  = xr[1]
    e1 = np.exp(-(y - params["y_min"]))
    e2 = np.exp(-(params["y_max"] - y))
    dhdy = (e1 - e2) / (e1 + e2)
    grad = np.array([0.0, dhdy, 0.0, 0.0])
    return cbf_boundary(xr), grad @ f_vec(xr), grad @ g_mat(xr)

# ─────────────────────────────────────────────────────────────────────────────
# SCBF Lie-derivative helper
#
# For B = exp(−γ h) the SCBF condition (Def 5 / paper Eq. 12) is:
#
#   ∂B/∂xr·(f+g u) + ∂B/∂xo·fo + σ  ≤  −aB + b
#
# ∂B/∂xr = −γB·∂h/∂xr,  ∂B/∂xo = −γB·∂h/∂xo
#
# Define (for the xo-augmented Lie derivative):
#   LfB_full = −γB·( LfH_xr  +  (∂h/∂xo)·fo )
#            = −γB·( LfH_xr  +  LfH_xo )
#   LgB      = −γB·LgH_xr
#
# The QP constraint becomes:
#   LgB·u − b  ≤  −LfB_full − σ − aB
#   −γB·LgH·u − b  ≤  γB·(LfH_xr + LfH_xo) − σ − aB
#
# In standard form  [row]·z ≤ rhs:
#   row = [γB·LgH[0],  γB·LgH[1],  0, 0, ..., −1, ...]
#   rhs =  γB·(LfH_xr + LfH_xo) − σ − aB
#
# (Signs: we want  LgB·u − b ≤ rhs_raw,
#         i.e.  −γB·LgH·u − b ≤ rhs_raw
#         i.e.  [γB·LgH, −1]·z ≤ −rhs_raw  ... careful below)
# ─────────────────────────────────────────────────────────────────────────────

def _scbf_row_rhs(LfH_xr, LfH_xo, LgH_xr, sigma, B0, a_scbf, gamma):
    """
    Returns (A_row_scaled, rhs) for the SCBF inequality:
        [γB·LgH_xr | −1] z  ≤  γB·(LfH_xr + LfH_xo) − σ + aB  ... wait

    Derivation (FIX Bug 3 & 6):
        ∂B/∂x·F_cl + σ ≤ −aB + b
        −γB(LfH_xr + LfH_xo)  −  γB·LgH_xr·u  +  σ  ≤  −aB + b
        −γB·LgH_xr·u  −  b  ≤  γB(LfH_xr + LfH_xo)  −  σ  +  aB  ... (*)

    Standard QP form:  A·z ≤ rhs  where z contains u and b:
        row for u-part  :  γB·LgH_xr   (positive)
        coeff of −b     :  −1
        rhs             :  γB·(LfH_xr + LfH_xo) − σ + aB

    Wait — (*) has the negatives on the left, let's be explicit.
    Let ℓ = LfH_xr + LfH_xo.  The inequality (*) says:
        −γB·LgH[0]·a  −  γB·LgH[1]·β  −  b  ≤  γB·ℓ − σ + aB
    Row coefficients for [a, β, ..., b_slot, ...] in z:
        [−γB·LgH[0], −γB·LgH[1],  0, 0, ...,  −1,  ...]
    rhs = γB·ℓ − σ + aB
    """
    gB     = gamma * B0
    ell    = LfH_xr + LfH_xo
    # A-row entries for u = [a, β]:
    A_u    = -gB * LgH_xr          # shape (2,)
    rhs    =  gB * ell - sigma + a_scbf * B0
    return A_u, rhs                 # caller places −1 for the b slot

# ─────────────────────────────────────────────────────────────────────────────
# ACC SCBF  (derivation doc §6)
#
# h_acc = (Xo − X) − T·v − (v0−v)²/(2 cd g) − 5
#
# ∂h/∂xr:  [−1, 0, dh_dv, 0]  where dh_dv = −T + (v0−v)/(cd·g)
#           (here v0 = lead_vehicle.speed, the leading car's speed)
# ∂h/∂xo:  [1, 0]    (Xo appears with +1)
#
# LfH_xr = grad_xr · f(xr)  =  −v·cos(ψ) + dh_dv·0  (f = [v cosψ, v sinψ, 0, 0])
# LfH_xo = (∂h/∂xo) · fo   =  1·vc  (fo = [vc, slope·vc], first component)
# LgH_xr = grad_xr · g(xr)  (2-vector)
# ─────────────────────────────────────────────────────────────────────────────
def cbf_acc(xr, lead_veh, p=params):
    X, _, v, _ = xr
    Xo  = lead_veh.position[0]
    v0  = lead_veh.speed                    # leading car speed (acts as v0 in IDM formula)
    cdg = p["cd"] * p["g"]

    h        = Xo - X - p["T"] * v - 0.5 * (v0 - v)**2 / cdg - 5.0
    dh_dv    = -p["T"] + (v0 - v) / cdg
    grad_xr  = np.array([-1.0, 0.0, dh_dv, 0.0])

    LfH_xr  = grad_xr @ f_vec(xr)
    LgH_xr  = grad_xr @ g_mat(xr)         # shape (2,)

    # FIX 2: use smoothed vc (nominal drift), not instantaneous speed
    vc       = get_vc_estimate(lead_veh)   # paper: vc,i — constant nominal velocity
    LfH_xo  = 1.0 * vc                    # (∂h/∂Xo)·vc = 1·vc

    sig = sigma_acc(h, lead_veh, p)
    return h, LfH_xr, LfH_xo, LgH_xr, sig


def cbf_target(xr, xi, yi, veh, p=params):
    """
    Target-ellipse SCBF  (derivation doc §8)

    h = (xi−X)²/Bx² + (yi−Y)²/By² − 1

    ∂h/∂xr = [−2(xi−X)/Bx², −2(yi−Y)/By², 0, 0]
    ∂h/∂xo = [+2(xi−X)/Bx², +2(yi−Y)/By²]   (xo is the obstacle position)
    LfH_xo = (∂h/∂xo) · fo_obs
    """
    X, Y, _, _ = xr
    dx = xi - X;  dy = yi - Y
    Bx, By = p["Bx"], p["By"]

    h       = dx**2 / Bx**2 + dy**2 / By**2 - 1.0
    grad_xr = np.array([-2.0*dx/Bx**2, -2.0*dy/By**2, 0.0, 0.0])
    LfH_xr  = grad_xr @ f_vec(xr)
    LgH_xr  = grad_xr @ g_mat(xr)         # shape (2,)

    # FIX Bug 4: include xo drift for target
    dh_dxo = np.array([2.0*dx/Bx**2, 2.0*dy/By**2])
    fo_obs = f_sto(veh)                    # [vc, slope·vc]
    LfH_xo = float(dh_dxo @ fo_obs)

    sig = sigma_target(h, xi, yi, xr, veh, p)
    return h, LfH_xr, LfH_xo, LgH_xr, sig

# ─────────────────────────────────────────────────────────────────────────────
# Vehicle finders
# ─────────────────────────────────────────────────────────────────────────────
def find_lead_vehicle(ego, vehicles):
    ego_lane = lane_index_from_y(ego.position[1])
    lead = None
    for veh in vehicles:
        if veh is ego:
            continue
        if (lane_index_from_y(veh.position[1]) == ego_lane
                and veh.position[0] > ego.position[0]):
            if lead is None or veh.position[0] < lead.position[0]:
                lead = veh
            
    return lead

def find_target_vehicle(ego, target_lane, vehicles):
    closest  = None
    min_dist = np.inf
    for veh in vehicles:
        if veh is ego:
            continue
        if lane_index_from_y(veh.position[1]) != target_lane:
            continue
        dist = np.hypot(veh.position[0] - ego.position[0],
                        veh.position[1] - ego.position[1])
        if dist < min_dist:
            min_dist = dist
            closest  = veh
    return closest


# FIX 5: collect ALL nearby vehicles (not just one) for multi-agent SCBF
def find_nearby_vehicles(ego, vehicles, p=params):
    """Return list of non-ego vehicles within proximity_radius.
    Paper (Sec. V): only agents within distance 3 are constrained each iteration."""
    nearby = []
    for veh in vehicles:
        if veh is ego:
            continue
        dist = np.hypot(veh.position[0] - ego.position[0],
                        veh.position[1] - ego.position[1])
        if dist <= p["proximity_radius"]:
            nearby.append(veh)
    return nearby


# ─────────────────────────────────────────────────────────────────────────────
# QP builder  (FIX 5 — multi-agent SCBF)
#
# Decision vector  z = [a, β, δ_v, δ_y, b_0, b_1, …, b_{M-1}]
#                       0   1    2    3   4    5        3+M
#
# Base size = 4 (u + slacks).  Each obstacle adds one b_i slot.
# ACC is obstacle index 0 when present; nearby obstacles follow.
#
# SCBF constraint for agent i:
#   row = [−γBᵢ·LgHᵢ[0], −γBᵢ·LgHᵢ[1], 0, 0, …, −1 (at col 4+i), …]
#   rhs =  γBᵢ·(LfHᵢ_xr + LfHᵢ_xo) − σᵢ + a·Bᵢ
# ─────────────────────────────────────────────────────────────────────────────
def build_qp(xr, u_ref, y_t, clf_alpha, clf_beta,
             lead_vehicle, nearby_vehicles, p=params):
    """
    Returns (P, q, A_ub, b_ub, diag) for
        min  ½ zᵀPz + qᵀz    s.t.  A_ub z ≤ b_ub

    FIX 5: handles M ≥ 0 obstacle agents (lead + nearby) simultaneously.
    Each agent gets its own b_i decision variable and risk bound.
    """
    a_scbf = p["a_scbf"]
    gamma  = p["gamma"]

    # ── Build ordered agent list: lead first, then other nearby ─────────────
    agents = []
    if lead_vehicle is not None:
        agents.append(("acc", lead_vehicle))
    for veh in nearby_vehicles:
        if veh is not lead_vehicle:
            agents.append(("obs", veh))
    M = len(agents)   # number of SCBF agents

    # z dimension: [a, β, δ_v, δ_y] + M b-variables
    n_z = 4 + M

    # ── CLF / boundary terms ─────────────────────────────────────────────────
    Vv, LfVv, LgVv = lie_clf_v(xr)
    Vy, LfVy, LgVy = lie_clf_y(xr, y_t)
    hB, LfHB, LgHB = lie_cbf_boundary(xr)

    # ── Cost matrix P ────────────────────────────────────────────────────────
    P_diag = (
        [2*p["w_a"], 2*p["w_beta"], 2*p["w_delta_v"], 2*p["w_delta_y"]]
        + [2*p["w_b"]] * M
    )
    P = np.diag(P_diag)

    # ── Cost vector q ────────────────────────────────────────────────────────
    q = np.zeros(n_z)
    q[0] = -2 * p["w_a"]    * u_ref[0]
    q[1] = -2 * p["w_beta"] * u_ref[1]

    A_list, b_list = [], []

    def _row(u_part, dv=0.0, dy=0.0, b_slot=None, b_val=0.0):
        """Build one constraint row of length n_z."""
        row = np.zeros(n_z)
        row[0] = u_part[0]; row[1] = u_part[1]
        row[2] = dv;        row[3] = dy
        if b_slot is not None:
            row[4 + b_slot] = b_val
        return row.tolist()

    # ── CLF speed ────────────────────────────────────────────────────────────
    A_list.append(_row(LgVv, dv=-1.0))
    b_list.append(-LfVv - clf_beta * p["clf_rate_v"] * Vv)

    # ── CLF lateral ──────────────────────────────────────────────────────────
    A_list.append(_row(LgVy, dy=-1.0))
    b_list.append(-LfVy - clf_alpha * p["clf_rate_y"] * Vy)

    # ── Boundary CBF (deterministic) ─────────────────────────────────────────
    A_list.append(_row(-LgHB))
    b_list.append(LfHB + p["cbf_k"] * hB)

    # ── Per-agent SCBF constraints ────────────────────────────────────────────
    agent_diags = []
    for i, (kind, veh) in enumerate(agents):
        xi_v = veh.position[0]
        yi_v = veh.position[1]

        if kind == "acc":
            h_i, LfH_xr, LfH_xo, LgH, sig = cbf_acc(xr, veh, p)
        else:
            h_i, LfH_xr, LfH_xo, LgH, sig = cbf_target(xr, xi_v, yi_v, veh, p)

        B0_i    = barrier_value(h_i, gamma)
        b_max_i = compute_b_max(B0_i, p)

        A_u, rhs = _scbf_row_rhs(LfH_xr, LfH_xo, LgH, sig, B0_i, a_scbf, gamma)

        # SCBF inequality: [A_u | … | −1 at slot i | …] z ≤ rhs
        A_list.append(_row(A_u, b_slot=i, b_val=-1.0))
        b_list.append(rhs)

        # Risk bound:  b_i ≤ b_max_i
        A_list.append(_row(np.zeros(2), b_slot=i, b_val=1.0))
        b_list.append(b_max_i)
        # b_i ≥ 0
        A_list.append(_row(np.zeros(2), b_slot=i, b_val=-1.0))
        b_list.append(0.0)

        agent_diags.append({
            "kind": kind, "veh": veh,
            "h": h_i, "B0": B0_i,
            "LfH": LfH_xr + LfH_xo, "LgH": LgH,
            "sig": sig, "b_max": b_max_i,
        })

    # ── Input limits ─────────────────────────────────────────────────────────
    A_list.append(_row([ 1.0, 0.0])); b_list.append( p["a_max"])
    A_list.append(_row([-1.0, 0.0])); b_list.append(-p["a_min"])
    A_list.append(_row([ 0.0, 1.0])); b_list.append( p["beta_max"])
    A_list.append(_row([ 0.0,-1.0])); b_list.append(-p["beta_min"])

    A_np = np.array(A_list, dtype=float)
    b_np = np.array(b_list, dtype=float)

    # ── Diag dict (backwards-compatible keys for logging) ────────────────────
    # ACC agent is always index 0 if present
    acc_d  = agent_diags[0] if (M > 0 and agents[0][0] == "acc") else None
    lane_d = None
    for d in agent_diags:
        if d["kind"] == "obs":
            lane_d = d; break

    diag = {
        "Vv": Vv, "Vy": Vy, "hB": hB, "LfHB": LfHB, "LgHB": LgHB,
        "h_acc":    acc_d["h"]   if acc_d  else np.inf,
        "LfH_acc":  acc_d["LfH"] if acc_d  else 0.0,
        "LgH_acc":  acc_d["LgH"] if acc_d  else np.zeros(2),
        "sig_acc":  acc_d["sig"] if acc_d  else 0.0,
        "B0_acc":   acc_d["B0"]  if acc_d  else 0.0,
        "b_max_acc":acc_d["b_max"] if acc_d else a_scbf,
        "h_lane":   lane_d["h"]   if lane_d else np.nan,
        "LfH_lane": lane_d["LfH"] if lane_d else np.nan,
        "LgH_lane": lane_d["LgH"] if lane_d else np.zeros(2),
        "sig_lane": lane_d["sig"] if lane_d else np.nan,
        "B0_lane":  lane_d["B0"]  if lane_d else np.nan,
        "b_max_lane":lane_d["b_max"] if lane_d else np.nan,
        # FIX 6: worst-case risk upper bound across all agents
        "agent_diags": agent_diags,
        "M": M,
        "n_z": n_z,
    }
    return csc_matrix(P), q, csc_matrix(A_np), b_np, diag


def build_qp_fallback(xr, u_ref, y_t, clf_alpha, clf_beta,
                      lead_vehicle, p=params):
    """
    Fallback QP: ACC constraint only (no nearby obstacles).
    z = [a, β, δ_v, δ_y, b_acc]   (5-vector)
    """
    lead_list = [lead_vehicle] if lead_vehicle is not None else []
    return build_qp(xr, u_ref, y_t, clf_alpha, clf_beta,
                    lead_vehicle, [], p)


# ─────────────────────────────────────────────────────────────────────────────
# Controller: solve QP with fallback chain
# ─────────────────────────────────────────────────────────────────────────────
def ctrl_qp(xr, u_ref, y_t, clf_alpha, clf_beta,
            lead_vehicle, nearby_vehicles, p=params):

    diag = None

    # Attempt 1: full multi-agent SCBF QP
    try:
        P, q, A, b, diag = build_qp(
            xr, u_ref, y_t, clf_alpha, clf_beta,
            lead_vehicle, nearby_vehicles, p)
        sol = solve_qp(P, q, A, b, solver="osqp")
        if sol is not None:
            return sol, diag
    except Exception as e:
        print(f"[QP1 failed] {e}")

    # Attempt 2: fallback — ACC constraint only
    try:
        P2, q2, A2, b2, diag = build_qp_fallback(
            xr, u_ref, y_t, clf_alpha, clf_beta,
            lead_vehicle, p)
        sol = solve_qp(P2, q2, A2, b2, solver="osqp")
        if sol is not None:
            n_full = diag["n_z"] if diag else 5
            padded = np.zeros(max(n_full, 5))
            padded[:len(sol)] = sol
            return padded, diag
    except Exception as e:
        print(f"[QP2 failed] {e}")

    # Attempt 3: zero control
    print("[WARN] Both QPs failed — zero control.")
    if diag is None:
        Vv, _, _ = lie_clf_v(xr)
        Vy, _, _ = lie_clf_y(xr, y_t)
        hB, LfHB, LgHB = lie_cbf_boundary(xr)
        diag = {
            "Vv": Vv, "Vy": Vy, "hB": hB, "LfHB": LfHB, "LgHB": LgHB,
            "h_acc": np.inf, "LfH_acc": 0.0, "LgH_acc": np.zeros(2),
            "sig_acc": 0.0, "B0_acc": 0.0, "b_max_acc": 0.0,
            "h_lane": np.nan, "LfH_lane": np.nan,
            "LgH_lane": np.zeros(2), "sig_lane": np.nan,
            "B0_lane": np.nan, "b_max_lane": np.nan,
            "agent_diags": [], "M": 0, "n_z": 4,
        }
    n_z = diag.get("n_z", 4)
    return np.zeros(max(n_z, 5)), diag




# ─────────────────────────────────────────────────────────────────────────────
# Pygame / Gym setup
# ─────────────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("SCBF Lane Control — Corrected")

pygame.joystick.init()
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick:", joystick.get_name())

env = gym.make("highway-v0", render_mode="human")
env = env.unwrapped
env.configure({
    "lanes_count":          params["N_lanes"],
    "vehicles_count":       50,
    "duration":             200,
    "simulation_frequency": 30,
    "policy_frequency":     30,
    "action":               {"type": "ContinuousAction"},
    "vehicles_density":     1.3,
})
obs, _ = env.reset()
ego    = env.vehicle

dt    = 1.0 / env.config["policy_frequency"]
steps = int(env.config["duration"] * env.config["policy_frequency"])

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logs = {k: [] for k in [
    "t", "x", "y", "v", "psi", "a", "beta",
    "Vv", "Vy", "hB", "LfHB", "LgHuB", "marginB",
    "h_acc", "LfH_acc", "LgHu_acc", "margin_acc", "sig_acc",
    "psi_dot", "beta_sat", "y_dot",
    "z_dist", "lead_present",
    "h_lane", "LfH_lane", "LgHu_lane", "sig_lane",
    "b_acc", "b_lane", "b_max_acc", "b_max_lane",
    "risk_upper_bound",   # pB = 1 − (1−B0)·exp(−b·T)   Eq. (10) paper
]}

# ─────────────────────────────────────────────────────────────────────────────
# Live plot setup
# ─────────────────────────────────────────────────────────────────────────────
plt.ion()
fig_live, axs_live = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_live.suptitle("SCBF Live Monitor", fontsize=13)

ax_risk, ax_acc, ax_lane = axs_live

ax_risk.set_ylabel("Risk upper bound")
ax_risk.set_title("Risk upper bound p̄_B  (≤ p_bar req.)")
ax_risk.axhline(params["p_bar"], ls="--", color="r",
                label=f"p_bar = {params['p_bar']}", lw=1)
ax_risk.legend(loc="upper right", fontsize=8)
ax_risk.grid(alpha=0.3)

ax_acc.set_ylabel("σ_acc")
ax_acc.set_title("SCBF_acc  stochastic correction σ")
ax_acc.grid(alpha=0.3)

ax_lane.set_ylabel("σ_lane")
ax_lane.set_title("SCBF lane  stochastic correction σ")
ax_lane.set_xlabel("Time (s)")
ax_lane.grid(alpha=0.3)

line_risk, = ax_risk.plot([], [], lw=2, color="orange", label="p̄_B")
line_acc,  = ax_acc.plot([], [],  lw=2, color="steelblue", label="σ_acc")
line_lane, = ax_lane.plot([], [], lw=2, color="seagreen",  label="σ_lane")

for ln, ax in [(line_risk, ax_risk), (line_acc, ax_acc), (line_lane, ax_lane)]:
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
fig_live.canvas.draw()
fig_live.canvas.flush_events()

LIVE_PLOT_INTERVAL = 10   # update every N steps (tune for speed vs. smoothness)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
beta_ref        = 0.0
joystick_mode   = 0
steering_unlock = False
prev_target_veh = None

for k in range(steps):
    # ── Event handling ────────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); env.close(); exit()

    if joystick is not None:
        pygame.event.pump()
        trig_left  = joystick.get_axis(4)
        trig_right = joystick.get_axis(5)
        if trig_left  != -1: joystick_mode = 1
        if trig_right != -1: joystick_mode = 0
        beta_ref = (joystick.get_axis(0) * params["beta_max"]
                    if joystick_mode == 1 else 0.0)
    else:
        keys = pygame.key.get_pressed()
        steering_unlock = bool(keys[pygame.K_SPACE])
        if steering_unlock:
            if keys[pygame.K_RIGHT]:
                beta_ref += 0.001
            elif keys[pygame.K_LEFT]:
                beta_ref -= 0.001
            else:
                beta_ref = 0.0

    beta_ref  = np.clip(beta_ref, params["beta_min"], params["beta_max"])
    clf_alpha = 0.4 if steering_unlock else 1.0
    clf_beta  = 0.3 if steering_unlock else 1.0

    # ── Ego state ─────────────────────────────────────────────────────────
    xr = np.array([ego.position[0], ego.position[1],
                   ego.speed, ego.heading])

    curr_lane = lane_index_from_y(ego.position[1])
    y_t       = lane_center_from_index(curr_lane)

    # ── Lead vehicle (ACC) ────────────────────────────────────────────────
    lead_vehicle = find_lead_vehicle(ego, env.road.vehicles)
    z_dist       = (lead_vehicle.position[0] - ego.position[0]
                    if lead_vehicle is not None else np.inf)
    lead_present = 1.0 if lead_vehicle is not None else 0.0

    # ── FIX 5: collect ALL nearby vehicles for multi-agent SCBF ──────────
    nearby_vehicles = find_nearby_vehicles(ego, env.road.vehicles, params)

    # ── Target lane / vehicle (for colour highlight only) ─────────────────
    target_lane = curr_lane
    if steering_unlock:
        if beta_ref < 0.0 and curr_lane > 0:
            target_lane = curr_lane - 1
        elif beta_ref > 0.0 and curr_lane < params["N_lanes"] - 1:
            target_lane = curr_lane + 1

    target_veh = find_target_vehicle(ego, target_lane, env.road.vehicles)

    # Colour management
    if prev_target_veh is not None:
        prev_target_veh.color = COLOR_DEFAULT
    if target_veh is not None:
        target_veh.color = COLOR_TARGET
        prev_target_veh  = target_veh
    else:
        prev_target_veh = None

    print(f"[k={k}] steer={steering_unlock}  agents={len(nearby_vehicles)}  "
          f"lead={'yes' if lead_vehicle else 'no'}")

    # ── Solve QP (multi-agent SCBF) ───────────────────────────────────────
    z, diag = ctrl_qp(
        xr, np.array([0.0, beta_ref]), y_t,
        clf_alpha, clf_beta,
        lead_vehicle, nearby_vehicles,
    )

    # ── Step environment ──────────────────────────────────────────────────
    obs, _, term, trunc, _ = env.step(z[:2])

    # FIX 3: Correct Wiener increment — N(0, dt), scaled only by go = cᵢ·[1,γ]
    # Removed the erroneous ×5 amplification.
    for veh in env.road.vehicles:
        if veh is ego:
            continue
        dw = np.random.normal(0.0, np.sqrt(dt))   # proper Wiener step: N(0,dt)
        go = g_sto(veh)                             # cᵢ·[1, γ_slope] — no ×5
        veh.position[0] += go[0] * dw
        veh.position[1] += go[1] * dw

    # ── Derived quantities for logging ────────────────────────────────────
    psi_dot  = xr[2] / params["l_r"] * z[1]
    LgHuB    = diag["LgHB"] @ z[:2]
    marginB  = diag["LfHB"] + LgHuB + params["cbf_k"] * diag["hB"]

    h_acc_val = diag["h_acc"]
    if lead_present > 0 and not np.isinf(h_acc_val):
        LgHu_acc   = diag["LgH_acc"] @ z[:2]
        margin_acc = diag["LfH_acc"] + LgHu_acc + params["cbf_k_acc"] * h_acc_val
    else:
        LgHu_acc = margin_acc = 0.0
        h_acc_val = 0.0

    # FIX 6: worst-case risk upper bound across ALL active agents
    # pB_i = 1 − (1−B0_i)·exp(−b_i·T)  [Eq. 10 paper]
    T = params["T"]
    risk_ub = 0.0
    agent_diags = diag.get("agent_diags", [])
    for i, ad in enumerate(agent_diags):
        b_i   = float(z[4 + i]) if (4 + i) < len(z) else 0.0
        B0_i  = float(ad["B0"])
        pB_i  = 1.0 - (1.0 - B0_i) * np.exp(-b_i * T)
        risk_ub = max(risk_ub, pB_i)
    # also keep legacy b_acc / b_lane for plots
    b_acc_sol  = float(z[4]) if len(z) > 4 else 0.0
    b_lane_sol = float(z[5]) if len(z) > 5 else 0.0

    # ── Logging ───────────────────────────────────────────────────────────
    logs["t"].append(k * dt)
    logs["x"].append(xr[0]);          logs["y"].append(xr[1])
    logs["v"].append(xr[2]);          logs["psi"].append(xr[3])
    logs["a"].append(z[0]);           logs["beta"].append(z[1])
    logs["Vv"].append(diag["Vv"]);    logs["Vy"].append(diag["Vy"])
    logs["hB"].append(diag["hB"])
    logs["LfHB"].append(diag["LfHB"])
    logs["LgHuB"].append(LgHuB);      logs["marginB"].append(marginB)
    logs["h_acc"].append(h_acc_val)
    logs["LfH_acc"].append(diag["LfH_acc"])
    logs["LgHu_acc"].append(LgHu_acc)
    logs["margin_acc"].append(margin_acc)
    logs["sig_acc"].append(diag["sig_acc"])
    logs["psi_dot"].append(psi_dot)
    logs["beta_sat"].append(abs(z[1]) / params["beta_max"])
    logs["y_dot"].append(xr[2] * np.sin(xr[3]))
    logs["z_dist"].append(z_dist if z_dist != np.inf else np.nan)
    logs["lead_present"].append(lead_present)
    logs["h_lane"].append(diag["h_lane"])
    logs["LfH_lane"].append(diag["LfH_lane"])

    lg_lane = diag["LgH_lane"]
    logs["LgHu_lane"].append(
        float(lg_lane @ z[:2])
        if not (np.isscalar(lg_lane) and np.isnan(lg_lane))
        else np.nan)

    logs["sig_lane"].append(diag["sig_lane"])
    logs["b_acc"].append(b_acc_sol)
    logs["b_lane"].append(b_lane_sol)
    logs["b_max_acc"].append(diag["b_max_acc"])
    logs["b_max_lane"].append(
        diag["b_max_lane"]
        if not np.isnan(diag.get("b_max_lane", np.nan))
        else np.nan)
    logs["risk_upper_bound"].append(risk_ub)

    # ── Live plot update ──────────────────────────────────────────────────
    if k % LIVE_PLOT_INTERVAL == 0 and len(logs["t"]) > 1:
        t_arr    = np.array(logs["t"],               dtype=float)
        risk_arr = np.array(logs["risk_upper_bound"], dtype=float)
        acc_arr  = np.array(logs["sig_acc"],          dtype=float)
        lane_arr = np.array(logs["sig_lane"],         dtype=float)

        line_risk.set_data(t_arr, risk_arr)
        line_acc.set_data(t_arr,  acc_arr)
        line_lane.set_data(t_arr, lane_arr)

        for ax, arr in [(ax_risk, risk_arr), (ax_acc, acc_arr), (ax_lane, lane_arr)]:
            if len(arr) > 0:
                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    ymin, ymax = finite.min(), finite.max()
                    pad = max((ymax - ymin) * 0.15, 1e-6)
                    ax.set_xlim(t_arr[0], max(t_arr[-1], t_arr[0] + 0.1))
                    ax.set_ylim(ymin - pad, ymax + pad)

        fig_live.canvas.draw()
        fig_live.canvas.flush_events()

    if term or trunc:
        break

env.close()
pygame.quit()

# ─────────────────────────────────────────────────────────────────────────────
# Final live-plot refresh — do one last full draw after simulation ends
# ─────────────────────────────────────────────────────────────────────────────
if len(logs["t"]) > 1:
    t_arr    = np.array(logs["t"],               dtype=float)
    risk_arr = np.array(logs["risk_upper_bound"], dtype=float)
    acc_arr  = np.array(logs["sig_acc"],          dtype=float)
    lane_arr = np.array(logs["sig_lane"],         dtype=float)

    line_risk.set_data(t_arr, risk_arr)
    line_acc.set_data(t_arr,  acc_arr)
    line_lane.set_data(t_arr, lane_arr)

    for ax, arr in [(ax_risk, risk_arr), (ax_acc, acc_arr), (ax_lane, lane_arr)]:
        if len(arr) > 0:
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                ymin, ymax = finite.min(), finite.max()
                pad = max((ymax - ymin) * 0.15, 1e-6)
                ax.set_xlim(t_arr[0], t_arr[-1])
                ax.set_ylim(ymin - pad, ymax + pad)

    fig_live.canvas.draw()
    fig_live.canvas.flush_events()
    plt.savefig("scbf_live_results.png", dpi=120)

plt.ioff()
plt.show()
