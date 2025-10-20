# gcode_to_time_series.py
# G-code -> uniformly resampled time series for printed path only
# New: option to trim to a layer window delimited by "; layer 1" ... "; layer end"
# Also recognizes formats like ";LAYER:1". Case-insensitive.

from __future__ import annotations
import re
import numpy as np
from typing import Tuple

def _strip_comment(line: str) -> str:
    # Remove ;... and (...) comments
    s = line.split(";")[0]
    out, paren = [], 0
    for ch in s:
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren-1)
        elif paren == 0:
            out.append(ch)
    return "".join(out).strip()

def _parse_float_after(prefix: str, token: str):
    if token.upper().startswith(prefix):
        try:
            return float(token[len(prefix):])
        except:
            return None
    return None

def _slice_to_layer_window(text: str, start_layer: int = 1, use_layer_markers: bool = True) -> str:
    """
    Keep only text between a 'start layer' marker (e.g., '; layer 1' or ';LAYER:1')
    and an 'end of layers' marker ('; layer end'). If not found, return original.
    """
    if not use_layer_markers:
        return text

    lines = text.splitlines()

    # Robust patterns (case-insensitive)
    pat_layer_num = re.compile(r";\s*layer[:\s]+(\d+)", re.IGNORECASE)  # "; layer 3" or ";LAYER:3"
    pat_layer1    = re.compile(r";\s*layer\s*1\b", re.IGNORECASE)
    pat_end       = re.compile(r";\s*layer\s*end\b", re.IGNORECASE)

    start_idx = None
    end_idx   = None

    for i, raw in enumerate(lines):
        s = raw.strip()
        if start_idx is None:
            m = pat_layer_num.search(s)
            if m and int(m.group(1)) >= start_layer:
                start_idx = i
            elif pat_layer1.search(s) and start_layer == 1:
                start_idx = i
        else:
            if pat_end.search(s):
                end_idx = i
                break

    if start_idx is None:
        return text  # fall back: markers not present
    if end_idx is None:
        end_idx = len(lines)  # until EOF

    return "\n".join(lines[start_idx:end_idx])

def _resample_polyline(t: np.ndarray, P: np.ndarray, seg_len: float):
    if len(P) < 2:
        return t[:1], P[:1]
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = float(s[-1])
    if L <= 1e-12:
        return t[:1], P[:1]
    s_new = np.arange(0.0, L + 1e-12, seg_len)
    if s_new[-1] < L - 1e-9:
        s_new = np.append(s_new, L)
    x = np.interp(s_new, s, P[:, 0])
    y = np.interp(s_new, s, P[:, 1])
    z = np.interp(s_new, s, P[:, 2])
    t_new = np.interp(s_new, s, t)
    return t_new, np.column_stack([x, y, z])

def gcode_to_resampled_series_from_bytes(
    gcode_bytes: bytes,
    layer_height_mm: float,
    seg_len_m: float,
    center_xy: bool = True,
    e_threshold: float = 1e-6,
    use_layer_markers: bool = True,
    start_layer: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Parse G-code and produce a uniformly-resampled time series along *extruding* paths,
    optionally restricted to the window ; layer <start> ... ; layer end.

    Returns: t_res[s], x_res[m], y_res[m], zc_res[m], total_length[m]
    """
    full_text = gcode_bytes.decode("utf-8", errors="ignore")
    text = _slice_to_layer_window(full_text, start_layer=start_layer, use_layer_markers=use_layer_markers)

    mm = 1e-3
    extrude_absolute = True  # M82 default
    lastE = 0.0
    last = {"X": 0.0, "Y": 0.0, "Z": 0.0, "F": None, "E": 0.0}
    t_acc = 0.0
    verts = []

    for raw in text.splitlines():
        # keep comments for marker slicing already done; strip here for coords
        line = _strip_comment(raw)
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0].upper()

        if cmd == "M82":
            extrude_absolute = True
            continue
        if cmd == "M83":
            extrude_absolute = False
            continue
        if cmd == "G92":
            for p in parts[1:]:
                v = _parse_float_after("E", p.upper())
                if v is not None:
                    lastE = float(v)
                    last["E"] = lastE
            continue

        if cmd not in ("G0", "G00", "G1", "G01"):
            continue

        coords = {}
        for p in parts[1:]:
            pu = p.upper()
            for axis in ("X", "Y", "Z", "F", "E"):
                v = _parse_float_after(axis, pu)
                if v is not None:
                    coords[axis] = v
                    break

        for k in ("X", "Y", "Z", "F", "E"):
            if k not in coords:
                coords[k] = last[k]

        dx = coords["X"] - last["X"]
        dy = coords["Y"] - last["Y"]
        dz = coords["Z"] - last["Z"]
        dist_mm = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        if coords["F"] is not None and coords["F"] > 0 and dist_mm > 0:
            t_acc += (dist_mm / coords["F"]) * 60.0  # mm/min → s

        if extrude_absolute:
            dE = float(coords["E"] - lastE)
            lastE = float(coords["E"])
        else:
            dE = float(coords["E"])
        is_extruding = dE > e_threshold

        verts.append((t_acc, coords["X"], coords["Y"], coords["Z"], lastE, is_extruding))
        last.update(coords)

    if not verts:
        raise ValueError("No printable moves found inside the selected layer window.")

    V = np.array(verts, dtype=object)
    t_all = V[:, 0].astype(float)
    x_mm  = V[:, 1].astype(float)
    y_mm  = V[:, 2].astype(float)
    z_mm  = V[:, 3].astype(float)
    is_ex = V[:, 5].astype(bool)

    if center_xy and len(x_mm) > 0:
        x_mm = x_mm - 0.5 * (float(x_mm.min()) + float(x_mm.max()))
        y_mm = y_mm - 0.5 * (float(y_mm.min()) + float(y_mm.max()))

    # Only extruding segments (between vertices i-1 -> i)
    seg_ex = is_ex.copy()
    seg_ex[0] = False
    idx_ex = np.where(seg_ex)[0]
    if len(idx_ex) == 0:
        raise ValueError("No extrusion found in the selected layer window.")

    # Build centroid Z in meters and split into extruding subpaths
    H_m  = float(layer_height_mm) * mm
    zc_m = (z_mm * mm) - 0.5 * H_m
    P_m  = np.column_stack([x_mm * mm, y_mm * mm, zc_m])

    t_res_list, P_res_list = [], []
    i, N = 1, len(t_all)
    while i < N:
        if seg_ex[i]:
            i0 = i - 1
            j = i
            while j + 1 < N and seg_ex[j + 1]:
                j += 1
            i1 = j
            t_seg = t_all[i0:i1 + 1]
            P_seg = P_m[i0:i1 + 1, :]
            t_r, P_r = _resample_polyline(t_seg, P_seg, seg_len=seg_len_m)
            t_res_list.append(t_r)
            P_res_list.append(P_r)
            i = j + 1
        else:
            i += 1

    if not t_res_list:
        raise ValueError("No extruding subpaths after trimming.")

    t_res = np.concatenate(t_res_list, axis=0)
    P_res = np.vstack(P_res_list)

    # Rebase time to 0 at first extrusion
    t_res = t_res - t_res[0]

    # Total printed length
    Ltot_m = 0.0
    for P_seg in P_res_list:
        if len(P_seg) >= 2:
            Ltot_m += float(np.sum(np.linalg.norm(P_seg[1:] - P_seg[:-1], axis=1)))

    return t_res, P_res[:, 0], P_res[:, 1], P_res[:, 2], Ltot_m
