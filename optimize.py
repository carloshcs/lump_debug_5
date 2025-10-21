# optimize.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
from model import GrowingLumpModel
from simulate import simulate_states


# ---------------- basic helpers ----------------
def layer_of(z: np.ndarray, H: float) -> np.ndarray:
    """Raw layer index from Z (0,1,2,...) based on height H."""
    return (np.floor(z / H + 1e-6)).astype(int)


# ---------------- layer info ----------------
def compute_layer_info_ordinal(times: np.ndarray,
                               x: np.ndarray, y: np.ndarray, zc: np.ndarray,
                               H: float) -> Dict:
    """
    Build an ordinal layer map:
      - ord_ids[i] in 1..K marks which ordinal layer segment i belongs to
      - starts: list of (L_ord, first_index_of_layer, time_at_first_index, raw_layer_id)
      - layer_time: list of (L_ord, span_seconds)
      - next_starts: for every layer that has a next layer:
            (L_prev_ord, index_of_next_start, time_of_next_start, x_next, y_next)
    """
    raw = layer_of(zc, H)

    order_map: Dict[int, int] = {}
    ord_ids = np.empty_like(raw)
    starts: List[Tuple[int, int, float, int]] = []
    k = 1
    for i, Lraw in enumerate(raw):
        if Lraw not in order_map:
            order_map[Lraw] = k
            starts.append((k, i, float(times[i]), int(Lraw)))
            k += 1
        ord_ids[i] = order_map[Lraw]

    layer_time: List[Tuple[int, float]] = []
    for L in range(1, k):
        idx = np.where(ord_ids == L)[0]
        if idx.size == 0:
            continue
        tspan = float(times[idx[-1]] - times[idx[0]])
        layer_time.append((L, tspan))

    next_starts: List[Tuple[int, int, float, float, float]] = []
    for j in range(1, len(starts)):
        Lprev, _, _, _ = starts[j - 1]
        _, idx_next, t_next, _ = starts[j]
        next_starts.append((Lprev, idx_next, float(t_next),
                            float(x[idx_next]), float(y[idx_next])))

    return dict(
        ord_ids=ord_ids,
        starts=starts,
        layer_time=layer_time,
        next_starts=next_starts
    )


# ---------------- extract substrate temps at next-start times ----------------
def compute_layer_pairs(states: List[Dict],
                        info: Dict, H: float):
    """
    For each tuple in next_starts (Lprev, idx_next, t_next, x0, y0):
      - Find snapshot nearest to t_next
      - On the previous layer (Lprev), compute:
          * mean temperature across that layer
          * point temperature closest (in XY) to (x0,y0)
    Returns:
      pairs_point: [(Lprev, T_point), ...]
      pairs_mean:  [(Lprev, T_mean), ...]
    """
    pairs_point, pairs_mean = [], []
    snap_times = np.array([s["time"] for s in states])

    def ord_from_raw(raw_layers, starts):
        map_raw_to_ord = {raw: L for (L, _, _, raw) in starts}
        return np.array([map_raw_to_ord.get(int(rr), None) for rr in raw_layers])

    for (Lprev, idx_next, t_next, x0, y0) in info["next_starts"]:
        sidx = int(np.argmin(np.abs(snap_times - t_next)))
        snap = states[sidx]
        P = snap["pos"]      # (N,3)
        TT = snap["T"]       # (N,)
        raw_layers = layer_of(P[:, 2], H)
        ord_layers = ord_from_raw(raw_layers, info["starts"])

        mask_prev = (ord_layers == Lprev)
        if np.any(mask_prev):
            T_mean = float(np.mean(TT[mask_prev])); pairs_mean.append((Lprev, T_mean))
            XY = P[mask_prev, :2]
            d = np.linalg.norm(XY - np.array([x0, y0]), axis=1)
            j = int(np.argmin(d))
            T_point = float(TT[mask_prev][j])
            pairs_point.append((Lprev, T_point))

    return pairs_point, pairs_mean


# ---------------- time scaling ----------------
def apply_speed_factors(times: np.ndarray,
                        ord_ids: np.ndarray,
                        speed_factors: np.ndarray) -> np.ndarray:
    """
    Scale the cumulative time sequence per segment using per-layer multipliers.
    ord_ids are 1-based ordinal layer IDs per segment.
    """
    times = np.asarray(times, dtype=float)
    ord_ids = np.asarray(ord_ids, dtype=int)
    t_new = np.zeros_like(times, dtype=float)
    t_new[0] = float(times[0])
    for i in range(1, len(times)):
        sL = float(speed_factors[int(ord_ids[i]) - 1])
        dt = float(times[i] - times[i - 1])
        t_new[i] = t_new[i - 1] + sL * dt
    return t_new


# ---------------- optimizer: per-layer bisection ----------------
def optimize_layer_speeds_sequential(
    cfg: Dict,
    target_T: float = 120.0,
    iters: int = 8,
    tol: float = 1.0,
    smin: float = 0.5,
    smax: float = 2.5,
    status_cb: Optional[Callable[[str], None]] = None
):
    """
    For each layer L that has a 'next start', choose time scale s[L] so that
    the substrate temperature at the next-layer start is close to target_T.

    Monotone rule (your requirement):
      - If measured T < target_T -> SPEED UP (decrease s)
      - If measured T > target_T -> SLOW DOWN (increase s)
    That is, temperature is approximately decreasing with s.

    Returns:
      s_opt, t_opt, info_opt, states_opt, logs, pp_opt, pm_opt
    """
    def status(msg: str):
        if status_cb:
            status_cb(msg)

    # Unpack
    t_base = cfg["tR"]; xR = cfg["xR"]; yR = cfg["yR"]; zcR = cfg["zcR"]
    ord_ids = cfg["ord_ids"]; H = cfg["H"]

    Nlayers = int(np.max(ord_ids)) if np.size(ord_ids) else 0
    s = np.ones(Nlayers, dtype=float)
    logs: List[str] = []

    # Which layers have a next start
    info0 = compute_layer_info_ordinal(t_base, xR, yR, zcR, H)
    valid_layers = sorted({int(Lprev) for (Lprev, _, _, _, _) in info0.get("next_starts", [])})

    # Cache: (L, s_value_rounded, tuple(prev_s_rounded)) -> temperature
    temp_cache: Dict[Tuple[int, float, Tuple[float, ...]], Optional[float]] = {}

    def eval_layer_T(L: int, s_value: float) -> Optional[float]:
        """
        Evaluate point substrate temperature for layer L at candidate scale s_value.
        Uses pairs_point metric (closest point at the next start).
        """
        key = (L, round(float(s_value), 4), tuple(np.round(s[:L-1], 4)))
        if key in temp_cache:
            return temp_cache[key]

        s_try = s.copy()
        s_try[L-1] = float(s_value)
        t_try = apply_speed_factors(t_base, ord_ids, s_try)

        # Recompute info, simulate, extract pairs
        info_try = compute_layer_info_ordinal(t_try, xR, yR, zcR, H)
        model = _build_model(cfg)

        try:
            record_times = [tt for (_, _, tt, _, _) in info_try.get("next_starts", [])]
        except Exception:
            record_times = None

        states_try, _ = simulate_states(
            model, t_try, xR, yR, zcR,
            dt=cfg["DT"], post_cooldown=cfg["COOLDOWN"],
            snap_interval=cfg["SNAP_INT"],
            record_times=record_times
        )
        pairs_point, _ = compute_layer_pairs(states_try, info_try, H)
        Tmap = {int(LL): float(T) for (LL, T) in pairs_point}
        T_L = Tmap.get(L, None)
        temp_cache[key] = T_L
        return T_L

    for L in valid_layers:
        lo, hi = float(smin), float(smax)  # lo=faster/hotter, hi=slower/cooler

        T_lo = eval_layer_T(L, lo)
        T_hi = eval_layer_T(L, hi)
        logs.append(f"[L{L}] bracket smin={lo:.3f} -> T={T_lo}, smax={hi:.3f} -> T={T_hi}")

        # If both invalid, keep default
        if T_lo is None and T_hi is None:
            logs.append(f"[L{L}] WARN: bracket temps unavailable; keep s=1.0")
            s[L-1] = 1.0
            continue

        # If only one end valid, choose it
        if T_lo is None and T_hi is not None:
            s[L-1] = hi
            logs.append(f"[L{L}] WARN: hot end invalid; choose slow s={hi:.3f} (T={T_hi})")
            continue
        if T_hi is None and T_lo is not None:
            s[L-1] = lo
            logs.append(f"[L{L}] WARN: cool end invalid; choose fast s={lo:.3f} (T={T_lo})")
            continue

        # Reachability clamp (remember: T decreases with s)
        if target_T >= T_lo:
            s[L-1] = lo
            logs.append(f"[L{L}] target >= hot end; clamp s={lo:.3f} (T={T_lo})")
            continue
        if target_T <= T_hi:
            s[L-1] = hi
            logs.append(f"[L{L}] target <= cool end; clamp s={hi:.3f} (T={T_hi})")
            continue

        # Proper bracket: T_lo > target_T > T_hi
        left_s, right_s = lo, hi
        left_T, right_T = T_lo, T_hi
        best_s, best_err = 1.0, 1e9
        best_T = None

        for k in range(max(1, int(iters))):
            mid_s = 0.5 * (left_s + right_s)
            mid_T = eval_layer_T(L, mid_s)

            if mid_T is None:
                # If mid failed, shrink toward a valid side
                if right_T is not None:
                    left_s = mid_s
                elif left_T is not None:
                    right_s = mid_s
                else:
                    break
                continue

            err = mid_T - target_T
            logs.append(f"[L{L}] iter {k+1}: s={mid_s:.4f} -> T={mid_T:.2f} (err={err:.2f})")

            if abs(err) < abs(best_err):
                best_err, best_s, best_T = err, mid_s, mid_T
            if abs(err) <= tol:
                break

            # Too hot => increase s (slower). Too cold => decrease s (faster).
            if err > 0.0:
                left_s, left_T = mid_s, mid_T
            else:
                right_s, right_T = mid_s, mid_T

        s[L-1] = float(best_s)

        status(f"Layer {L}: chosen s={s[L-1]:.3f}, T~{best_T if best_T is not None else 'n/a'}")

    # Final schedule and one full simulation
    t_opt = apply_speed_factors(t_base, ord_ids, s)
    info_opt = compute_layer_info_ordinal(t_opt, xR, yR, zcR, H)

    model_opt = _build_model(cfg)

    try:
        record_times_opt = [tt for (_, _, tt, _, _) in info_opt.get("next_starts", [])]
    except Exception:
        record_times_opt = None

    states_opt, logs_opt = simulate_states(
        model_opt, t_opt, xR, yR, zcR,
        dt=cfg["DT"], post_cooldown=cfg["COOLDOWN"],
        snap_interval=cfg["SNAP_INT"],
        record_times=record_times_opt
    )
    pp_opt, pm_opt = compute_layer_pairs(states_opt, info_opt, H)

    logs_out = []
    if isinstance(logs_opt, list):
        logs_out.extend([str(x) for x in logs_opt])
    else:
        logs_out.append(str(logs_opt))

    return s, t_opt, info_opt, states_opt, logs_out, pp_opt, pm_opt


# ---------------- write optimized G-code ----------------
# --- in optimize.py ---

def write_optimized_gcode(gcode_bytes: bytes, speed_factors: np.ndarray) -> bytes:
    """
    Return a new G-code where each printing move's feedrate F is scaled by 1/s[L]
    inside that layer L. Travel moves (no E) are left as-is.

    Recognizes layer markers:
      ; layer N
      ;LAYER:N
      ;LAYER: N
    """
    text = gcode_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    out: List[str] = []

    # Regex helpers
    rx_layer_a = re.compile(r";\s*layer\s+(\d+)", re.IGNORECASE)   # "; layer 12"
    rx_layer_b = re.compile(r";\s*LAYER:\s*(\d+)", re.IGNORECASE)  # ";LAYER:12"
    rx_f = re.compile(r"(\s|^)F(\d+(\.\d*)?)", re.IGNORECASE)      # feedrate
    rx_e = re.compile(r"(\s|^)E(-?\d+(\.\d*)?)", re.IGNORECASE)    # extrusion
    rx_g1 = re.compile(r"^\s*G1(\s|$)", re.IGNORECASE)             # G1 line

    # Track current layer and last seen F (to rescale if needed)
    cur_layer = None
    last_F = None

    # Header comment block with the chosen scales
    out.append("; OPT: ---- per-layer time/scales ----")
    for i, s in enumerate(speed_factors, start=1):
        out.append(f"; OPT: layer {i} time_x={s:.3f} speed_x={1.0/max(1e-12, s):.3f}")
    out.append("; OPT: --------------------------------")

    for line in lines:
        # Detect a layer start
        mA = rx_layer_a.search(line)
        mB = rx_layer_b.search(line)
        if mA or mB:
            cur_layer = int((mA or mB).group(1))
            sL = float(speed_factors[cur_layer - 1]) if 1 <= cur_layer <= len(speed_factors) else 1.0
            vL = 1.0 / max(1e-12, sL)

            # Annotate layer start + optionally set a new modal F for this layer
            out.append(f"; OPT: enter layer {cur_layer} (time_x={sL:.3f} -> speed_x={vL:.3f})")
            # If we have a known modal F, set a new modal F for the layer based on it
            if last_F is not None:
                out.append(f"G1 F{last_F * vL:.2f}")
            out.append(line)
            continue

        # Normal line handling
        if rx_g1.match(line):
            # Track modal F if provided
            mF = rx_f.search(line)
            if mF:
                try:
                    last_F = float(mF.group(2))
                except Exception:
                    pass

            # Scale printing moves (those with E value) by 1/sL
            if cur_layer is not None and rx_e.search(line):
                sL = float(speed_factors[cur_layer - 1]) if 1 <= cur_layer <= len(speed_factors) else 1.0
                vL = 1.0 / max(1e-12, sL)

                if mF:
                    # rewrite existing F inline
                    try:
                        Fval = float(mF.group(2))
                        newF = max(1.0, Fval * vL)
                        line = rx_f.sub(lambda m: f"{m.group(1)}F{newF:.2f}", line, count=1)
                        last_F = newF
                    except Exception:
                        pass
                else:
                    # no F on this G1 — inject one at the end using modal * vL if known
                    if last_F is not None:
                        newF = max(1.0, last_F * vL)
                        line = line.rstrip() + f" F{newF:.2f}"
                        last_F = newF

        out.append(line)

    return ("\n".join(out) + "\n").encode("utf-8")

def _build_model(cfg) -> GrowingLumpModel:
    return GrowingLumpModel(
        seg_len=cfg["SEG_LEN"],
        seg_width=cfg["SEG_WIDTH"],
        seg_height=cfg["H"],
        rho=cfg["RHO"], cp=cfg["CP"], k=cfg["K"],
        T_bed=cfg["T_BED"], T_amb=cfg["T_INF"], h=cfg["H_COEF"],
        dt=cfg["DT"], post_cooldown=cfg["COOLDOWN"],
        link_max=cfg["LINK_MAX"], vert_radius=cfg["V_RAD"],
        T_nozzle=cfg["T_NOZZLE"],
        bed_contact_frac=cfg["BED_FRAC"],
        contact_area_frac=cfg["CONTACT_FRAC"],
        # NEW:
        microgrid_enable = cfg.get("MICROGRID_ENABLE", False),
        nx = int(cfg.get("MG_NX", 1)),
        nz = int(cfg.get("MG_NZ", 1)),
        use_tabular=cfg.get("USE_TABULAR", False),
        k_table=cfg.get("K_TABLE"),
        cp_table=cfg.get("CP_TABLE"),
    )
