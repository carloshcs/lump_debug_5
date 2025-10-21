import os
import numpy as np
import streamlit as st
from material_db import MATERIALS, load_all_materials


def parse_table_text(text: str):
    T, V = [], []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2:
            try:
                T.append(float(parts[0]))
                V.append(float(parts[1]))
            except ValueError:
                pass
    return (T, V) if T else None


def get_inputs(S, key_prefix="main"):
    """
    Build all sidebar input sections.
    key_prefix ensures unique widget keys if called multiple times.
    """
    st.sidebar.header("Inputs")

    # === Upload ===
    gcode_file = st.sidebar.file_uploader(
        "Upload G-code",
        type=["gcode", "txt", "nc"],
        key=f"{key_prefix}_gcode_uploader"
    )

    # === Printing parameters ===
    with st.sidebar.expander("🧵 Printing parameters", expanded=False):
        SEG_WIDTH_mm = st.number_input("Bead width W (mm)", 0.5, 50.0, 6.0, 0.5, format="%.1f", key=f"{key_prefix}_SEG_WIDTH_mm")
        SEG_HEIGHT_mm = st.number_input("Bead height H (mm)", 0.5, 10.0, 1.5, 0.5, format="%.1f", key=f"{key_prefix}_SEG_HEIGHT_mm")
        T_NOZZLE = st.number_input("Nozzle temp (°C)", 40.0, 500.0, 250.0, 5.0, key=f"{key_prefix}_T_NOZZLE")
        T_BED = st.number_input("Bed temp (°C)", 0.0, 200.0, 80.0, 5.0, key=f"{key_prefix}_T_BED")
        T_INF = st.number_input("Ambient temp (°C)", -40.0, 100.0, 20.0, 1.0, key=f"{key_prefix}_T_INF")
        H_COEF = st.number_input("Convection h (W/m²·K)", 1e-8, 50.0, 5.0, 0.5, key=f"{key_prefix}_H_COEF")
        HINT_SPEED_MM_S = st.number_input(
            "Estimated print speed (mm/s) — used only for Δt hint",
            1.0, 200.0, float(S.get("HINT_SPEED_MM_S", 20.0)), 1.0,
            key=f"{key_prefix}_HINT_SPEED_MM_S"
        )
        S["HINT_SPEED_MM_S"] = HINT_SPEED_MM_S

    # === Material properties ===
    MATERIALS_DIR = os.path.abspath("materials_db")
    os.makedirs(MATERIALS_DIR, exist_ok=True)
    MATERIALS.update(load_all_materials(MATERIALS_DIR))
    mat_names = sorted(MATERIALS.keys())

    with st.sidebar.expander("🧪 Material properties", expanded=False):
        if not mat_names:
            st.warning("No materials found. Add a new material below.")
            mat_choice = None
        else:
            mat_choice = st.selectbox("Material", mat_names, index=0, key=f"{key_prefix}_mat_choice")

        def _seed_from(name):
            d = MATERIALS.get(name, {}) if name else {}
            def _get(k, default):
                try:
                    return float(d.get(k, default))
                except Exception:
                    return default
            return (
                _get("rho", 1200.0),
                _get("k", 0.25),
                _get("cp", 1300.0),
                _get("emissivity", 0.85),
                bool(d.get("enable_radiation", True)),
            )

        rho_seed, k_seed, cp_seed, emi_seed, rad_seed = _seed_from(mat_choice)

        last_mat_key = f"{key_prefix}_last_material"
        prev_material = st.session_state.get(last_mat_key)
        mat_changed = mat_choice != prev_material
        st.session_state[last_mat_key] = mat_choice

        def _mark_inputs_dirty():
            if getattr(S, "has_base", False):
                st.session_state.inputs_dirty = True

        tab_key = f"{key_prefix}_USE_TABULAR"
        if tab_key not in st.session_state:
            st.session_state[tab_key] = bool(S.get("USE_TABULAR", False))

        def _on_tabular_change():
            S["USE_TABULAR"] = bool(st.session_state[tab_key])
            _mark_inputs_dirty()

        USE_TABULAR = st.checkbox(
            "Use temperature-dependent k(T), cp(T)",
            key=tab_key,
            on_change=_on_tabular_change,
        )
        USE_TABULAR = bool(USE_TABULAR)
        S["USE_TABULAR"] = USE_TABULAR

        rad_key = f"{key_prefix}_ENABLE_RADIATION"
        if mat_changed:
            st.session_state[rad_key] = bool(rad_seed)
        elif rad_key not in st.session_state:
            st.session_state[rad_key] = bool(S.get("ENABLE_RADIATION", rad_seed))

        def _on_radiation_change():
            S["ENABLE_RADIATION"] = bool(st.session_state[rad_key])
            _mark_inputs_dirty()

        ENABLE_RADIATION = st.checkbox(
            "Enable radiation losses",
            key=rad_key,
            on_change=_on_radiation_change,
        )
        ENABLE_RADIATION = bool(ENABLE_RADIATION)
        S["ENABLE_RADIATION"] = ENABLE_RADIATION

        def readonly(label, val):
            st.markdown(
                f"<div style='background-color:#f7f7f7; padding:6px; border-radius:4px; color:#333; margin-bottom:4px;'>{label}: <b>{val}</b></div>",
                unsafe_allow_html=True
            )

        if mat_choice:
            readonly("Density ρ (kg/m³)", rho_seed)
            readonly("Conductivity k (W/m·K)", k_seed)
            readonly("Specific heat cₚ (J/kg·K)", cp_seed)
            readonly("Emissivity ε (–)", emi_seed)
            RHO, K, CP, EMISSIVITY = rho_seed, k_seed, cp_seed, emi_seed
        else:
            RHO, K, CP, EMISSIVITY = 1200.0, 0.25, 1300.0, 0.85
        S["EMISSIVITY"] = EMISSIVITY

        # --- Simplified single button ---
        if st.button("🧩 Add / Edit Material", key=f"{key_prefix}_edit_material"):
            S.mode = "editor"
            S.show_editor = True
            S.editor_target = mat_choice
            st.rerun()


    # === Calculation & Simulation ===
    with st.sidebar.expander("🧮 Calculation method & simulation", expanded=False):
        COOLDOWN = st.number_input(
            "Post-cooldown (s)", 0.0, 300.0, 0.0, 5.0, key=f"{key_prefix}_COOLDOWN"
        )

        noob_key = f"{key_prefix}_NOOB_SETUP"
        noob_default = bool(S.get("NOOB_SETUP", True))
        noob_setup = st.checkbox(
            "Noob setup",
            value=noob_default,
            key=noob_key,
            help="Keeps the recommended defaults. Uncheck to fine-tune advanced parameters.",
        )
        S["NOOB_SETUP"] = bool(noob_setup)
        disable_adv = bool(noob_setup)

        SEG_LEN_mm = st.number_input(
            "Lump length (mm)", 0.1, 100.0, 10.0, 1.0,
            key=f"{key_prefix}_SEG_LEN_mm",
            disabled=disable_adv,
        )
        BED_FRAC = st.slider(
            "Bed contact fraction (layer 0)", 0.0, 1.0, 1.0, 0.05,
            key=f"{key_prefix}_BED_FRAC",
            disabled=disable_adv,
        )
        CONTACT_FR = st.slider(
            "Layer–layer contact fraction", 0.0, 1.0, 1.0, 0.05,
            key=f"{key_prefix}_CONTACT_FR",
            disabled=disable_adv,
        )

        tseg = None
        src = ""
        try:
            cache = getattr(st.session_state, "cache", None)
            if isinstance(cache, dict):
                t_raw = cache.get("t")
                if t_raw is not None:
                    t_arr = np.asarray(t_raw, dtype=float)
                    if t_arr.size > 1:
                        dt_segs = np.diff(t_arr)
                        dt_segs = dt_segs[dt_segs > 1e-6]
                        if dt_segs.size:
                            tseg = float(np.median(dt_segs))
            if tseg is None:
                v_user = float(S.get("HINT_SPEED_MM_S", 20.0))
                tseg = SEG_LEN_mm / max(1e-6, v_user)
                src = f"(using your speed {v_user:.0f} mm/s)"
            else:
                src = "(from loaded G-code)"
        except Exception:
            tseg = None

        if not tseg:
            tseg = max(SEG_LEN_mm / max(1e-6, float(S.get("HINT_SPEED_MM_S", 20.0))), 1e-3)
            if not src:
                src = "(using hint speed)"

        lo, hi = 0.10 * tseg, 0.50 * tseg
        recommended_dt = float(np.clip(round(hi, 3), 0.001, 1.0))
        dt_key = f"{key_prefix}_DT"
        if disable_adv or dt_key not in st.session_state:
            st.session_state[dt_key] = recommended_dt
        DT = st.number_input(
            "dt (s)", 0.001, 1.0, step=0.001, format="%.3f",
            key=dt_key,
            disabled=disable_adv,
        )
        st.caption(
            f"For the current lump size, one-segment time ≈ {tseg:.3f} s {src}. "
            f"Try Δt in [{lo:.3f}, {hi:.3f}] s."
        )

        SNAP_INT = st.number_input(
            "Snapshot every (s)", 0.1, 60.0, 5.0, 0.5,
            key=f"{key_prefix}_SNAP_INT",
            disabled=disable_adv,
        )
        LINK_MAX_F = st.number_input(
            "Horizontal link × SEG_LEN", 1.0, 3.0, 1.10, 0.05,
            key=f"{key_prefix}_LINK_MAX_F",
            disabled=disable_adv,
        )
        V_RAD_MM = st.text_input(
            "Vertical search radius (mm, blank = 0.6×W)",
            "",
            key=f"{key_prefix}_V_RAD_MM",
            disabled=disable_adv,
        )
        MARKER_SIZE = st.slider(
            "Marker size (GIF)", 5, 60, 18,
            key=f"{key_prefix}_MARKER_SIZE",
            disabled=disable_adv,
        )

        use_mg = st.checkbox(
            "Enable microgrid",
            value=bool(S.get("MICROGRID_ENABLE", False)),
            key=f"{key_prefix}_MICROGRID_ENABLE",
            disabled=disable_adv,
        )
        mg_nz = st.number_input(
            "Nz", 1, 50, int(S.get("MG_NZ", 5)), 1,
            disabled=(disable_adv or not use_mg),
            key=f"{key_prefix}_MG_NZ",
        )
        mg_nx = st.number_input(
            "Nx", 1, 100, int(S.get("MG_NX", 20)), 1,
            disabled=(disable_adv or not use_mg),
            key=f"{key_prefix}_MG_NX",
        )
        S["MICROGRID_ENABLE"], S["MG_NZ"], S["MG_NX"] = bool(use_mg), int(mg_nz), int(mg_nx)

    # Track configuration signature to detect when cached simulation results become stale.
    signature = (
        ("gcode_name", getattr(gcode_file, "name", "")),
        ("material", mat_choice or ""),
        ("rho", round(float(RHO), 6)),
        ("k", round(float(K), 6)),
        ("cp", round(float(CP), 6)),
        ("emissivity", round(float(EMISSIVITY), 6)),
        ("use_tabular", bool(USE_TABULAR)),
        ("enable_radiation", bool(ENABLE_RADIATION)),
        ("seg_width_mm", round(float(SEG_WIDTH_mm), 6)),
        ("seg_height_mm", round(float(SEG_HEIGHT_mm), 6)),
        ("seg_len_mm", round(float(SEG_LEN_mm), 6)),
        ("t_nozzle", round(float(T_NOZZLE), 6)),
        ("t_bed", round(float(T_BED), 6)),
        ("t_inf", round(float(T_INF), 6)),
        ("h_coef", round(float(H_COEF), 6)),
        ("bed_frac", round(float(BED_FRAC), 6)),
        ("contact_fr", round(float(CONTACT_FR), 6)),
        ("dt", round(float(DT), 6)),
        ("cooldown", round(float(COOLDOWN), 6)),
        ("snap_int", round(float(SNAP_INT), 6)),
        ("link_max_f", round(float(LINK_MAX_F), 6)),
        ("v_rad_mm", V_RAD_MM.strip()),
        ("marker_size", int(MARKER_SIZE)),
        ("microgrid_enable", bool(use_mg)),
        ("mg_nz", int(mg_nz)),
        ("mg_nx", int(mg_nx)),
    )
    S.current_signature = signature
    last_sig = S.get("last_sim_signature")
    if getattr(S, "has_base", False) and last_sig is not None:
        if signature != last_sig:
            S.inputs_dirty = True
        else:
            S.inputs_dirty = False
    elif not getattr(S, "has_base", False):
        S.inputs_dirty = False
    else:
        S.inputs_dirty = False

    return {
        "gcode_file": gcode_file,
        "SEG_WIDTH_mm": SEG_WIDTH_mm, "SEG_HEIGHT_mm": SEG_HEIGHT_mm,
        "T_NOZZLE": T_NOZZLE, "T_BED": T_BED, "T_INF": T_INF, "H_COEF": H_COEF,
        "RHO": RHO, "K": K, "CP": CP, "EMISSIVITY": EMISSIVITY,
        "USE_TABULAR": USE_TABULAR, "ENABLE_RADIATION": ENABLE_RADIATION,
        "SEG_LEN_mm": SEG_LEN_mm, "BED_FRAC": BED_FRAC, "CONTACT_FR": CONTACT_FR,
        "DT": DT, "COOLDOWN": COOLDOWN, "SNAP_INT": SNAP_INT,
        "LINK_MAX_F": LINK_MAX_F, "V_RAD_MM": V_RAD_MM,
        "MARKER_SIZE": MARKER_SIZE, "MATERIALS_DIR": MATERIALS_DIR,
        "MATERIALS": MATERIALS, "parse_table_text": parse_table_text
    }
