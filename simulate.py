# -*- coding: utf-8 -*-
"""
simulate.py
Time-stepping driver: integrates the GrowingLumpModel and returns snapshots.
Supports optional microgrid (thickness model), convection, and radiation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional


# -------------------------------------------------------------------------
# Optional intra-lump microgrid (1-D through thickness)
# -------------------------------------------------------------------------
class MicrogridUpdater:
    def __init__(self, seg_width, seg_height, rho, cp, k, h, T_amb, T_bed,
                 bed_contact_frac=0.3, contact_area_frac=0.5, nx=1, nz=1,
                 # convection multipliers
                 h_top_mul: float = 0.0, h_side_mul: float = 1.0, h_bottom_mul: float = 1.0,
                 # radiation controls (mirrors convection face-multipliers)
                 emissivity: float = 0.8, enable_radiation: bool = True,
                 rad_top_mul: float = 0.0, rad_side_mul: float = 1.0, rad_bottom_mul: float = 0.0):
        self.nx = max(1, int(nx))
        self.nz = max(1, int(nz))
        self.seg_w = float(seg_width)
        self.seg_h = float(seg_height)

        self.rho = float(rho)
        self.cp = float(cp)
        self.k = float(k)
        self.h = float(h)

        self.T_amb = float(T_amb)
        self.T_bed = float(T_bed)

        self.bed_frac = float(bed_contact_frac)
        self.contact_frac = float(contact_area_frac)

        # convection multipliers
        self.h_top_mul = float(h_top_mul)
        self.h_side_mul = float(h_side_mul)
        self.h_bottom_mul = float(h_bottom_mul)

        # radiation params
        self.emissivity = float(emissivity)
        self.enable_radiation = bool(enable_radiation)
        self.rad_top_mul = float(rad_top_mul)
        self.rad_side_mul = float(rad_side_mul)
        self.rad_bottom_mul = float(rad_bottom_mul)  # kept for completeness (not used in BC)

        self.sigma = 5.670374419e-8  # W/m^2/K^4

        self.alpha = self.k / (self.rho * self.cp + 1e-12)
        self.dz = self.seg_h / self.nz if self.nz > 0 else 1.0
        # explicit stability guidance: dt <= dz^2/(2*alpha)
        self.dt_stable = (self.dz ** 2) / (2.0 * self.alpha + 1e-12) if self.alpha > 0 else 1e9

    def init_grid(self, n_lumps: int, T0=None):
        T0 = self.T_amb if T0 is None else float(T0)
        return np.full((int(n_lumps), self.nz, self.nx), T0, dtype=float)

    @staticmethod
    def avg(grid: np.ndarray) -> np.ndarray:
        if grid.size == 0:
            return np.zeros((0,), dtype=float)
        return grid.mean(axis=(1, 2))

    @staticmethod
    def apply_uniform_delta(grid: np.ndarray, dT_vec: np.ndarray):
        if grid.size == 0 or dT_vec is None:
            return
        grid += np.asarray(dT_vec, dtype=float).reshape((-1, 1, 1))

    def step_1d_thickness(self, grid: np.ndarray, dt: float,
                          on_bed_mask: Optional[np.ndarray] = None,
                          top_exposed_frac: Optional[np.ndarray] = None):
        """
        Explicit 1-D FD through thickness.

        BCs:
          - Bottom: convection (scaled by h_bottom_mul). **No radiation at bottom.**
          - Top: convection (scaled by h_top_mul and top_exposed_frac),
                 + radiation to ambient (Kelvin), gated by enable_radiation, emissivity,
                 rad_top_mul and top_exposed_frac.
          - No side radiation here (handled at lump level).
        """
        L = int(grid.shape[0])
        Nz = int(self.nz)
        if L == 0 or Nz <= 0:
            return

        lam = self.alpha * dt / (self.dz * self.dz + 1e-12)

        if on_bed_mask is None:
            on_bed_mask = np.zeros((L,), dtype=bool)
        else:
            on_bed_mask = np.asarray(on_bed_mask, dtype=bool).reshape(-1)

        if top_exposed_frac is None:
            top_exposed_frac = np.ones((L,), dtype=float)
        else:
            top_exposed_frac = np.clip(np.asarray(top_exposed_frac, dtype=float).reshape(-1), 0.0, 1.0)

        # Bottom convection effective h (boost if on bed), then scaled by h_bottom_mul
        h_bed_eff = np.where(on_bed_mask, self.h * np.clip(self.bed_frac, 0.0, 1.0) * 5.0, self.h)
        h_bed_eff = h_bed_eff * self.h_bottom_mul
        T_bot_env = np.where(on_bed_mask, self.T_bed, self.T_amb)
        Bi_bot_base = (h_bed_eff * self.dz) / (self.k + 1e-12)

        # Top convection Bi (scaled per lump by exposure)
        Bi_top_base = (self.h * self.h_top_mul * self.dz) / (self.k + 1e-12)

        # Radiation constants (Kelvin)
        T_ambK = self.T_amb + 273.15

        for i in range(L):
            Ti = grid[i, :, 0].copy()

            Bi_bot_i = float(Bi_bot_base[i])
            Bi_top_i = float(Bi_top_base) * float(top_exposed_frac[i])

            if Nz == 1:
                # Single node
                Ti0 = Ti[0]

                # Convection terms
                conv_top = -Bi_top_i * (Ti0 - self.T_amb)
                conv_bot = -Bi_bot_i * (Ti0 - T_bot_env[i])

                # Top radiation only
                if self.enable_radiation and self.emissivity > 0.0 and self.rad_top_mul > 0.0 and top_exposed_frac[i] > 0.0:
                    Ti0K = Ti0 + 273.15
                    rad_top = (2.0 * lam *
                               (self.emissivity * self.sigma *
                                self.rad_top_mul * top_exposed_frac[i] *
                                ((Ti0K ** 4) - (T_ambK ** 4)) * self.dz / (self.k + 1e-12)))
                else:
                    rad_top = 0.0

                Ti0 += 2.0 * lam * (conv_top + conv_bot) - rad_top
                Ti[0] = Ti0

            else:
                Tn = Ti.copy()

                # Interior nodes
                for j in range(1, Nz - 1):
                    Tn[j] = Ti[j] + lam * (Ti[j + 1] - 2.0 * Ti[j] + Ti[j - 1])

                # Bottom boundary: convection only (no radiation)
                j = 0
                Tn[j] = Ti[j] + 2.0 * lam * (Ti[j] - Ti[j])  # no-op part kept for structure clarity
                Tn[j] = Ti[j] + 2.0 * lam * (Ti[j + 1] - Ti[j]) \
                        - 2.0 * lam * Bi_bot_i * (Ti[j] - T_bot_env[i])

                # Top boundary: convection + (optional) radiation
                j = Nz - 1
                conv_top = -2.0 * lam * Bi_top_i * (Ti[j] - self.T_amb)

                if self.enable_radiation and self.emissivity > 0.0 and self.rad_top_mul > 0.0 and top_exposed_frac[i] > 0.0:
                    TijK = Ti[j] + 273.15
                    rad_top = (2.0 * lam *
                               (self.emissivity * self.sigma *
                                self.rad_top_mul * top_exposed_frac[i] *
                                ((TijK ** 4) - (T_ambK ** 4)) * self.dz / (self.k + 1e-12)))
                else:
                    rad_top = 0.0

                Tn[j] = Ti[j] + 2.0 * lam * (Ti[j - 1] - Ti[j]) + conv_top - rad_top

                Ti = Tn

            # write back (replicate through Nx copies)
            grid[i, :, :] = Ti.reshape((Nz, 1))


# -------------------------------------------------------------------------
def _snapshot(model, t_now: float, override_T: Optional[np.ndarray] = None) -> Dict:
    P = np.array(model.pos, float) if len(model.pos) else np.zeros((0, 3), float)
    if override_T is None:
        T = np.array(model.T, float) if len(model.T) else np.zeros((0,), float)
    else:
        T = np.array(override_T, float)
    return {"time": float(t_now), "pos": P, "T": T}


# -------------------------------------------------------------------------
def simulate_states(model, t, x, y, zc, dt=0.05, post_cooldown=0.0,
                    snap_interval=5.0, record_times=None):
    use_mg = bool(getattr(model, "microgrid_enable", False)) and \
             (int(getattr(model, "nz", 1)) > 1 or int(getattr(model, "nx", 1)) > 1)
    if use_mg:
        return _simulate_states_with_microgrid(model, t, x, y, zc, dt,
                                               post_cooldown,
                                               snap_interval=snap_interval,
                                               record_times=record_times)
    return _simulate_states_core(model, t, x, y, zc, dt,
                                 post_cooldown,
                                 snap_interval=snap_interval,
                                 record_times=record_times)


# -------------------------------------------------------------------------
def _simulate_states_core(model, t, x, y, zc, dt=0.05,
                          post_cooldown=0.0, snap_interval=5.0,
                          record_times=None):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    zc = np.asarray(zc, float)
    states: List[Dict] = []
    logs = []
    t_cur = 0.0
    i_event = 0
    t_last = float(t[-1]) if len(t) else 0.0
    t_limit = t_last + float(max(0.0, post_cooldown))
    next_snap = 0.0
    states.append(_snapshot(model, t_cur))
    forced = sorted(list(set([float(tt) for tt in (record_times or []) if tt >= 0.0])))

    while True:
        t_next_event = t[i_event] if i_event < len(t) else np.inf
        t_next_forced = forced[0] if forced else np.inf
        t_target = min(t_next_event, t_next_forced, t_limit)
        if t_cur >= t_limit - 1e-12:
            break
        while t_cur + 1e-12 < t_target:
            dt_step = min(dt, t_target - t_cur)
            model.step(float(dt_step))
            t_cur += float(dt_step)
            while t_cur + 1e-12 >= next_snap:
                states.append(_snapshot(model, next_snap))
                next_snap += float(snap_interval)
        while i_event < len(t) and t[i_event] <= t_cur + 1e-12:
            model.add_lump([x[i_event], y[i_event], zc[i_event]],
                           T_init=getattr(model, "T_nozzle", 300.0))
            i_event += 1
        if forced and abs(forced[0] - t_cur) <= 1e-12:
            states.append(_snapshot(model, t_cur))
            forced.pop(0)
        if t_cur >= t_limit - 1e-12 and states[-1]["time"] < t_limit - 1e-12:
            states.append(_snapshot(model, t_limit))
    if states and abs(states[-1]["time"] - t_limit) > 1e-9:
        states.append(_snapshot(model, t_limit))
    logs.append(
        f"simulate_states: events={len(t)}, duration={t_limit:.2f}s, dt={dt}, "
        f"cooldown={post_cooldown}, radiation={'on' if getattr(model,'enable_radiation',False) else 'off'}, "
        f"emissivity={getattr(model,'emissivity',0.0):.2f}, "
        f"tabular_props={'on' if getattr(model, 'use_tabular', False) else 'off'}"
    )
    return states, "\n".join(logs)


# -------------------------------------------------------------------------
def _simulate_states_with_microgrid(model, t, x, y, zc, dt=0.05,
                                    post_cooldown=0.0, snap_interval=5.0,
                                    record_times=None):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    zc = np.asarray(zc, float)
    states: List[Dict] = []
    logs = []
    t_cur = 0.0
    i_event = 0
    t_last = float(t[-1]) if len(t) else 0.0
    t_limit = t_last + float(max(0.0, post_cooldown))
    next_snap = 0.0

    mg = MicrogridUpdater(
        seg_width=getattr(model, "W", getattr(model, "seg_width", 1.0)),
        seg_height=getattr(model, "H", getattr(model, "seg_height", 1.0)),
        rho=getattr(model, "rho", 1000.0),
        cp=getattr(model, "cp", 1000.0),
        k=getattr(model, "k", 0.2),
        h=getattr(model, "h", 10.0),
        T_amb=getattr(model, "T_amb", 25.0),
        T_bed=getattr(model, "T_bed", 60.0),
        bed_contact_frac=getattr(model, "bed_contact_frac", 0.3),
        contact_area_frac=getattr(model, "contact_area_frac", 0.5),
        nx=int(getattr(model, "nx", 1)),
        nz=int(getattr(model, "nz", 1)),
        h_top_mul=getattr(model, "h_top_mul", 0.0),
        h_side_mul=getattr(model, "h_side_mul", 1.0),
        h_bottom_mul=getattr(model, "h_bottom_mul", 1.0),
        emissivity=getattr(model, "emissivity", 0.8),
        enable_radiation=getattr(model, "enable_radiation", True),
        rad_top_mul=getattr(model, "rad_top_mul", 0.0),
        rad_side_mul=getattr(model, "rad_side_mul", 1.0),
        rad_bottom_mul=getattr(model, "rad_bottom_mul", 0.0),
    )

    logs.append(
        "BC: h_top_mul={:.2f}, h_side_mul={:.2f}, h_bottom_mul={:.2f}; "
        "rad_top_mul={:.2f}, rad_side_mul={:.2f}, rad_bottom_mul={:.2f}; "
        "radiation={}, emissivity={:.2f}".format(
            mg.h_top_mul, mg.h_side_mul, mg.h_bottom_mul,
            mg.rad_top_mul, mg.rad_side_mul, mg.rad_bottom_mul,
            "on" if mg.enable_radiation else "off", mg.emissivity
        )
    )

    grid = mg.init_grid(0, T0=mg.T_amb)
    states.append(_snapshot(model, t_cur, override_T=MicrogridUpdater.avg(grid[:0])))

    forced = sorted(list(set([float(tt) for tt in (record_times or []) if tt >= 0.0])))

    eps = 1e-12
    while True:
        t_next_event = t[i_event] if i_event < len(t) else np.inf
        t_next_forced = forced[0] if forced else np.inf
        t_target = min(t_next_event, t_next_forced, t_limit)
        if t_cur >= t_limit - eps:
            break

        while t_cur + eps < t_target:
            dt_step = float(min(dt, t_target - t_cur))
            sub_dt = min(dt_step, mg.dt_stable)
            n_sub = max(1, int(np.ceil(dt_step / max(sub_dt, 1e-12))))
            n_sub = min(n_sub, 200)
            sub = dt_step / n_sub
            L = len(model.T)
            if grid.shape[0] < L:
                add = mg.init_grid(L - grid.shape[0], T0=mg.T_amb)
                grid = np.concatenate([grid, add], axis=0)

            for _ in range(n_sub):
                if L > 0:
                    # on-bed detection (lowest z's layer)
                    z_vals = np.array([p[2] for p in model.pos], dtype=float)
                    z0 = float(np.min(z_vals)) if L else 0.0
                    on_bed = z_vals <= (z0 + 0.5 * mg.seg_h)

                    # top exposure: use model.top_cover if present
                    if hasattr(model, "top_cover") and len(model.top_cover) >= L:
                        top_exposed = 1.0 - np.clip(np.array(model.top_cover[:L], dtype=float), 0.0, 1.0)
                    else:
                        top_exposed = np.ones((L,), dtype=float)

                    mg.step_1d_thickness(grid[:L, :, :], sub,
                                         on_bed_mask=on_bed,
                                         top_exposed_frac=top_exposed)

                    T_avg = MicrogridUpdater.avg(grid[:L, :, :])
                    model.T = [float(v) for v in T_avg]
                else:
                    T_avg = np.zeros((0,), dtype=float)

                # network/external step from model (includes side radiation at lump level)
                model.step(sub)

                # inject uniform ΔT back into grid
                if L > 0:
                    T_after = np.array(model.T, dtype=float)
                    dT_net = T_after - T_avg
                    MicrogridUpdater.apply_uniform_delta(grid[:L, :, :], dT_net)
                    model.T = [float(v) for v in MicrogridUpdater.avg(grid[:L, :, :])]

                t_cur += sub
                while t_cur + eps >= next_snap:
                    Lsnap = len(model.T)
                    T_snap = MicrogridUpdater.avg(grid[:Lsnap, :, :]) if Lsnap > 0 else np.zeros((0,), float)
                    states.append(_snapshot(model, next_snap, override_T=T_snap))
                    next_snap += float(snap_interval)

        # place new lumps due now
        while i_event < len(t) and t[i_event] <= t_cur + eps:
            model.add_lump([x[i_event], y[i_event], zc[i_event]],
                           T_init=getattr(model, "T_nozzle", 300.0))
            Lnew = len(model.T)
            if grid.shape[0] < Lnew:
                add = mg.init_grid(Lnew - grid.shape[0], T0=mg.T_amb)
                grid = np.concatenate([grid, add], axis=0)
            if Lnew > 0:
                T_init = float(getattr(model, "T_nozzle", mg.T_amb))
                grid[Lnew - 1, :, :] = T_init
                model.T[Lnew - 1] = T_init
            i_event += 1

        if forced and abs(forced[0] - t_cur) <= eps:
            Lsnap = len(model.T)
            T_snap = MicrogridUpdater.avg(grid[:Lsnap, :, :]) if Lsnap > 0 else np.zeros((0,), float)
            states.append(_snapshot(model, t_cur, override_T=T_snap))
            forced.pop(0)

    if states and abs(states[-1]["time"] - t_limit) > 1e-9:
        Lsnap = len(model.T)
        T_snap = MicrogridUpdater.avg(grid[:Lsnap, :, :]) if Lsnap > 0 else np.zeros((0,), float)
        states.append(_snapshot(model, t_limit, override_T=T_snap))

    logs.append(
        f"simulate_states[microgrid]: events={len(t)}, duration={t_limit:.2f}s, "
        f"dt={dt}, cooldown={post_cooldown}, mg.nz={mg.nz}, mg.nx={mg.nx}, "
        f"dt_stable≈{mg.dt_stable:.3g}s, radiation={'on' if mg.enable_radiation else 'off'}"
    )
    return states, "\n".join(logs)
