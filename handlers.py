# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:59:30 2025
@author: saunders
"""

# handlers.py
import os
from typing import Optional
import streamlit as st
import numpy as np

from gcode_to_time_series import gcode_to_resampled_series_from_bytes
from path_io import compute_bounds
from model import GrowingLumpModel
from simulate import simulate_states
from optimize import (
    compute_layer_info_ordinal, compute_layer_pairs,
    optimize_layer_speeds_sequential, write_optimized_gcode
)
from viz import (
    render_frames_from_states, build_gif_from_frames,
    last_frame_path, make_pdf_report,
)
from app_utils import mm_to_m, purge_frames

# ---------------- Run (base) ----------------
def run_base(
    gbytes: bytes,
    SEG_LEN_mm: float, SEG_WIDTH_mm: float, SEG_HEIGHT_mm: float,
    RHO: float, CP: float, K: float,
    T_NOZZLE: float, T_BED: float, T_INF: float, H_COEF: float,
    BED_FRAC: float, CONTACT_FR: float,
    DT: float, COOLDOWN: float, SNAP_INT: float,
    LINK_MAX_F: float, V_RAD_MM: str,
    MARKER_SIZE: int,
    # --- Radiation ---
    EMISSIVITY: float = 0.8,
    ENABLE_RADIATION: bool = False,
    S=None, ui=None, GIFS=None, DIRS=None, logs_slot=None, pdf_slot=None
):
    seg_len, seg_width, seg_height = map(mm_to_m, (SEG_LEN_mm, SEG_WIDTH_mm, SEG_HEIGHT_mm))
    S.cache["gbytes"] = gbytes

    # Parse and resample G-code
    t, x, y, zc, Ltot = gcode_to_resampled_series_from_bytes(
        gbytes,
        layer_height_mm=float(SEG_HEIGHT_mm),
        seg_len_m=float(seg_len),
        center_xy=True,
        e_threshold=1e-6,
        use_layer_markers=True
    )
    bounds = compute_bounds(x, y, zc)
    S.cache.update(dict(t=t, x=x, y=y, zc=zc, bounds=bounds, Ltot=f"{Ltot:.3f}"))

    # Layer info
    info = compute_layer_info_ordinal(t, x, y, zc, seg_height)
    S.cache["info"] = info

    # Optional vertical radius
    v_rad = None
    if V_RAD_MM.strip():
        try:
            v_rad = mm_to_m(float(V_RAD_MM))
        except Exception:
            v_rad = None

    # Microgrid enable flags
    microgrid_enable = bool(st.session_state.get("MICROGRID_ENABLE", False))
    mg_nx = int(st.session_state.get("MG_NX", 1))
    mg_nz = int(st.session_state.get("MG_NZ", 1))

    # Build model
    model = GrowingLumpModel(
        seg_len=seg_len, seg_width=seg_width, seg_height=seg_height,
        rho=RHO, cp=CP, k=K,
        T_bed=T_BED, T_amb=T_INF, h=H_COEF,
        dt=DT, post_cooldown=COOLDOWN,
        link_max=LINK_MAX_F * seg_len, vert_radius=v_rad,
        T_nozzle=T_NOZZLE,
        bed_contact_frac=BED_FRAC, contact_area_frac=CONTACT_FR,
        microgrid_enable=microgrid_enable, nx=mg_nx, nz=mg_nz,
        # --- Radiation ---
        emissivity=EMISSIVITY,
        enable_radiation=ENABLE_RADIATION
    )

    # Run simulation
    states, logs = simulate_states(
        model, t, x, y, zc,
        dt=DT, post_cooldown=COOLDOWN,
        snap_interval=SNAP_INT,
        record_times=[tt for (_, _, tt, _, _) in info.get("next_starts", [])]
    )
    S.cache["states"] = states
    logs_slot.code("\n".join(str(x) for x in logs) if isinstance(logs, list) else str(logs))

    # Extract substrate temps
    pairs_point, pairs_mean = compute_layer_pairs(states, info, seg_height)
    S.cache["pairs_point"], S.cache["pairs_mean"] = pairs_point, pairs_mean

    # ---- Colormap ----
    CMAP = st.session_state.get("CMAP", "turbo")

    # Render base GIFs
    for d in DIRS["base"].values():
        purge_frames(d)
    for p in GIFS["base"].values():
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    render_frames_from_states(states, bounds, DIRS["base"]["ISO"], "iso",
                              vmin=T_INF, vmax=T_NOZZLE, marker_size=MARKER_SIZE,
                              dpi=240, clean=True, cmap=CMAP)
    render_frames_from_states(states, bounds, DIRS["base"]["TOP"], "top",
                              vmin=T_INF, vmax=T_NOZZLE, marker_size=MARKER_SIZE,
                              dpi=240, clean=True, cmap=CMAP)
    render_frames_from_states(states, bounds, DIRS["base"]["FRONT"], "front",
                              vmin=T_INF, vmax=T_NOZZLE, marker_size=MARKER_SIZE,
                              dpi=240, clean=True, cmap=CMAP)

    build_gif_from_frames(DIRS["base"]["ISO"], GIFS["base"]["ISO"])
    build_gif_from_frames(DIRS["base"]["TOP"], GIFS["base"]["TOP"])
    build_gif_from_frames(DIRS["base"]["FRONT"], GIFS["base"]["FRONT"])

    # Mark new replay tick so the results page refreshes GIFs
    S.replay_tick += 1

    # Optional PDF
    base_last_pngs = [
        last_frame_path(DIRS["base"]["ISO"]),
        last_frame_path(DIRS["base"]["TOP"]),
        last_frame_path(DIRS["base"]["FRONT"]),
    ]
    pdf_path = os.path.join("outputs", "report_base.pdf")
    pdf_made = make_pdf_report(pdf_path, base_last_pngs,
                               title="Thermal simulation (base)",
                               notes=logs if isinstance(logs, str) else "")
    if pdf_made and os.path.exists(pdf_made):
        pdf_slot.download_button("Download page PDF (base)",
                                 data=open(pdf_made, "rb").read(),
                                 file_name="thermal_report_base.pdf",
                                 mime="application/pdf")

    S.has_base = True
    S.last_sim_signature = S.get("current_signature")
    S.inputs_dirty = False


# ---------------- Optimize ----------------
def optimize_to_target(
    opt_target: float, opt_iters: int, opt_tol: float, opt_smin: float, opt_smax: float,
    SEG_LEN_mm: float, SEG_WIDTH_mm: float, SEG_HEIGHT_mm: float,
    RHO: float, CP: float, K: float,
    T_NOZZLE: float, T_BED: float, T_INF: float, H_COEF: float,
    BED_FRAC: float, CONTACT_FR: float,
    DT: float, COOLDOWN: float, SNAP_INT: float,
    LINK_MAX_F: float, V_RAD_MM: str,
    MARKER_SIZE: int,
    # --- Radiation ---
    EMISSIVITY: float = 0.8,
    ENABLE_RADIATION: bool = False,
    S=None, ui=None, GIFS=None, DIRS=None, logs_slot=None
):
    # Pre-draw so UI does not flicker
    ui.draw_base_charts(S)
    ui.show_gifs("base", S, last_frame_path)
    ui.draw_opt_overlay_if_available(S)

    seg_len, seg_width, seg_height = map(mm_to_m, (SEG_LEN_mm, SEG_WIDTH_mm, SEG_HEIGHT_mm))

    cfg = dict(
        tR=S.cache["t"], xR=S.cache["x"], yR=S.cache["y"], zcR=S.cache["zc"],
        ord_ids=S.cache["info"]["ord_ids"],
        SEG_LEN=seg_len, SEG_WIDTH=seg_width, H=seg_height,
        RHO=RHO, CP=CP, K=K, T_BED=T_BED, T_INF=T_INF, H_COEF=H_COEF,
        DT=DT, COOLDOWN=COOLDOWN, SNAP_INT=SNAP_INT,
        LINK_MAX=LINK_MAX_F * seg_len,
        V_RAD=None if not V_RAD_MM.strip() else mm_to_m(float(V_RAD_MM)),
        T_NOZZLE=T_NOZZLE, BED_FRAC=BED_FRAC, CONTACT_FRAC=CONTACT_FR,
        MICROGRID_ENABLE=bool(st.session_state.get("MICROGRID_ENABLE", False)),
        MG_NX=int(st.session_state.get("MG_NX", 1)),
        MG_NZ=int(st.session_state.get("MG_NZ", 1)),
        # --- Radiation ---
        EMISSIVITY=EMISSIVITY,
        ENABLE_RADIATION=ENABLE_RADIATION,
    )

    s_opt, t_opt, info_opt, states_opt, logs_opt, pp_opt, pm_opt = optimize_layer_speeds_sequential(
        cfg, target_T=float(opt_target), iters=int(opt_iters),
        tol=float(opt_tol), smin=float(opt_smin), smax=float(opt_smax)
    )

    # Cache + logs
    S.cache["s_opt"] = s_opt
    S.cache["t_opt"] = t_opt
    S.cache["info_opt"] = info_opt
    S.cache["pp_opt"], S.cache["pm_opt"] = pp_opt, pm_opt
    logs_slot.code("\n".join(str(x) for x in logs_opt) if isinstance(logs_opt, list) else str(logs_opt))

    # Overlay and visuals
    if ui.sub_overlay_slot is not None:
        ui.sub_overlay_slot.empty()
    S.overlay_tick += 1
    ui.draw_opt_overlay_if_available(S)

    # ---- Colormap ----
    CMAP = st.session_state.get("CMAP", "turbo")

    # Render optimized GIFs
    bounds = S.cache["bounds"]
    for d in DIRS["opt"].values():
        purge_frames(d)
    for p in GIFS["opt"].values():
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    render_frames_from_states(states_opt, bounds, DIRS["opt"]["ISO"], "iso",
                              vmin=T_INF, vmax=T_NOZZLE, marker_size=MARKER_SIZE,
                              dpi=240, clean=True, cmap=CMAP)
    render_frames_from_states(states_opt, bounds, DIRS["opt"]["TOP"], "top",
                              vmin=T_INF, vmax=T_NOZZLE, marker_size=MARKER_SIZE,
                              dpi=240, clean=True, cmap=CMAP)
    render_frames_from_states(states_opt, bounds, DIRS["opt"]["FRONT"], "front",
                              vmin=T_INF, vmax=T_NOZZLE, marker_size=MARKER_SIZE,
                              dpi=240, clean=True, cmap=CMAP)
    build_gif_from_frames(DIRS["opt"]["ISO"], GIFS["opt"]["ISO"])
    build_gif_from_frames(DIRS["opt"]["TOP"], GIFS["opt"]["TOP"])
    build_gif_from_frames(DIRS["opt"]["FRONT"], GIFS["opt"]["FRONT"])

    ui.clear_opt_gif_slots()
    S.replay_tick += 1
    ui.show_gifs("opt", S, last_frame_path)

    # Build optimized G-code for download
    if "gbytes" in S.cache:
        S.cache["g_opt"] = write_optimized_gcode(S.cache["gbytes"], s_opt)
