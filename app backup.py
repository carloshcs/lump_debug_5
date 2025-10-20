import os
import numpy as np
import streamlit as st
from app_utils import build_output_dirs, init_state, mm_to_m
from ui import UI
from handlers import run_base, optimize_to_target
from viz import last_frame_path

# --- material DB loader ---
try:
    from material_db import MATERIALS, load_all_materials, parse_table_text
except Exception:
    MATERIALS = {}
    def load_all_materials(dir_path="materials_db"):
        mats = {}
        if not os.path.isdir(dir_path):
            return mats
        for fn in os.listdir(dir_path):
            if fn.lower().endswith(".txt"):
                name = fn[:-4]
                mats[name] = {}
                with open(os.path.join(dir_path, fn), "r", encoding="utf-8") as f:
                    text = f.read()
                section = None
                for raw in text.splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1].strip().lower()
                        if section == "k_table":
                            mats[name]["k_table"] = ([], [])
                        elif section == "cp_table":
                            mats[name]["cp_table"] = ([], [])
                        continue
                    if section in ("k_table", "cp_table"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == 2:
                            try:
                                T = float(parts[0])
                                V = float(parts[1])
                                mats[name][section][0].append(T)
                                mats[name][section][1].append(V)
                            except ValueError:
                                pass
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        mats[name][k.strip()] = v.strip()
        return mats

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

# --- Ensure folder exists ---
MATERIALS_DIR = os.path.abspath("materials_db")
os.makedirs(MATERIALS_DIR, exist_ok=True)

st.set_page_config(page_title="Lumped Thermal + Optimization", layout="wide")
st.title("Lumped Thermal Simulator (per-layer speed optimization)")

# ===== Sidebar: Inputs =====
st.sidebar.header("Inputs")
gcode_file = st.sidebar.file_uploader("Upload G-code", type=["gcode", "txt", "nc"])

# --- Printing parameters ---
with st.sidebar.expander("🧵 Printing parameters", expanded=False):
    SEG_WIDTH_mm = st.number_input("Bead width W (mm)", min_value=0.5, value=6.0, step=0.5, format="%.1f")
    SEG_HEIGHT_mm = st.number_input("Bead height H (mm)", min_value=0.5, value=1.5, step=0.5, format="%.1f")
    T_NOZZLE = st.number_input("Nozzle temp (°C)", min_value=40.0, value=250.0, step=5.0, format="%.1f")
    T_BED = st.number_input("Bed temp (°C)", min_value=0.0, value=80.0, step=5.0, format="%.1f")
    T_INF = st.number_input("Ambient temp (°C)", min_value=-40.0, value=20.0, step=1.0, format="%.1f")
    H_COEF = st.number_input("Convection h (W/m²·K)", min_value=0.00000001, value=5.0, step=0.5, format="%.1f")

    HINT_SPEED_MM_S = st.number_input(
        "Estimated print speed (mm/s) — used only for Δt hint",
        min_value=1.0,
        value=float(st.session_state.get("HINT_SPEED_MM_S", 20.0)),
        step=1.0,
        format="%.0f",
    )
    st.session_state["HINT_SPEED_MM_S"] = float(HINT_SPEED_MM_S)

# --- Material properties ---
with st.sidebar.expander("🧪 Material properties", expanded=False):
    MATERIALS.update(load_all_materials(MATERIALS_DIR))
    mat_names = sorted(MATERIALS.keys())

    if not mat_names:
        st.warning("No materials found. Add a new material below.")
        mat_choice = None
    else:
        mat_choice = st.selectbox("Material", mat_names, index=0)

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

    USE_TABULAR = st.checkbox(
        "Use temperature-dependent k(T), cp(T)",
        value=bool(st.session_state.get("USE_TABULAR", False)),
        help="Enable to use tabular data if defined for this material.",
    )
    ENABLE_RADIATION = st.checkbox(
        "Enable radiation losses",
        value=bool(st.session_state.get("ENABLE_RADIATION", rad_seed)),
        help="Include radiative losses using material emissivity.",
    )
    st.session_state["USE_TABULAR"] = USE_TABULAR
    st.session_state["ENABLE_RADIATION"] = ENABLE_RADIATION

    def readonly_input(label, value):
        st.markdown(
            f"<div style='background-color:#f0f0f0; padding:6px; border-radius:4px; color:#333;'>{label}: <b>{value}</b></div>",
            unsafe_allow_html=True,
        )

    if mat_choice:
        readonly_input("Density ρ (kg/m³)", rho_seed)
        readonly_input("Conductivity k (W/m·K)", k_seed)
        readonly_input("Specific heat cₚ (J/kg·K)", cp_seed)
        readonly_input("Emissivity ε (–)", emi_seed)
        RHO, K, CP, EMISSIVITY = rho_seed, k_seed, cp_seed, emi_seed
    else:
        RHO, K, CP, EMISSIVITY = 1200.0, 0.25, 1300.0, 0.85
    st.session_state["EMISSIVITY"] = EMISSIVITY

    if "show_editor" not in st.session_state:
        st.session_state["show_editor"] = False
        st.session_state["editor_mode"] = None
        st.session_state["editor_target"] = None

    cA, cB = st.columns(2)
    if cA.button("➕ Add new material"):
        st.session_state["show_editor"] = True
        st.session_state["editor_mode"] = "add"
    if cB.button("✏️ Edit selected", disabled=(mat_choice not in MATERIALS)):
        st.session_state["show_editor"] = True
        st.session_state["editor_mode"] = "edit"
        st.session_state["editor_target"] = mat_choice

# --- Popup Material Editor (centered) ---
if st.session_state.get("show_editor", False):
    st.markdown(
        """
        <style>
        .overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.45);
            z-index: 999;
        }
        .modal {
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: #fff;
            border-radius: 12px;
            padding: 30px;
            width: 480px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 25px rgba(0,0,0,0.3);
            z-index: 1000;
        }
        </style>
        <div class="overlay"></div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="modal">', unsafe_allow_html=True)
        mode = st.session_state.get("editor_mode", "add")
        target = st.session_state.get("editor_target", None)
        existing = MATERIALS.get(target, {}) if mode == "edit" and target else {}

        def pairs_to_text(pairs):
            if not pairs:
                return ""
            T, V = pairs
            return "\n".join(f"{t},{v}" for t, v in zip(T, V))

        name_default = existing.get("name", target or "")
        rho_default = float(existing.get("rho", 1200.0))
        k_default = float(existing.get("k", 0.25))
        cp_default = float(existing.get("cp", 1300.0))
        emi_default = float(existing.get("emissivity", 0.85))
        rad_default = bool(existing.get("enable_radiation", True))
        k_text_default = pairs_to_text(existing.get("k_table"))
        cp_text_default = pairs_to_text(existing.get("cp_table"))

        st.markdown("### ✏️ Material Editor")
        with st.form("mat_editor_popup", clear_on_submit=False):
            name_new = st.text_input("Name", value=name_default)
            rho_new = st.number_input("ρ (kg/m³)", value=rho_default, step=10.0)
            k_new = st.number_input("k (W/m·K)", value=k_default, step=0.01, format="%.3f")
            cp_new = st.number_input("cp (J/kg·K)", value=cp_default, step=10.0)
            rad_new = st.checkbox("Enable radiation", value=rad_default)
            emi_new = st.number_input("Emissivity ε (–)", min_value=0.0, max_value=1.0, value=emi_default, step=0.05)
            st.caption("Optional tables — format: `T,value` per line")
            k_table_text = st.text_area("k(T) table", value=k_text_default, height=100)
            cp_table_text = st.text_area("cp(T) table", value=cp_text_default, height=100)

            f1, f2 = st.columns(2)
            save_btn = f1.form_submit_button("💾 Save")
            close_btn = f2.form_submit_button("❌ Close")

            if save_btn:
                safe_name = name_new.strip().replace(" ", "_") or "Unnamed"
                k_pairs = parse_table_text(k_table_text)
                cp_pairs = parse_table_text(cp_table_text)
                path = os.path.join(MATERIALS_DIR, f"{safe_name}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"name = {safe_name}\n")
                    f.write(f"rho = {rho_new}\n")
                    f.write(f"k = {k_new}\n")
                    f.write(f"cp = {cp_new}\n")
                    f.write(f"emissivity = {emi_new}\n")
                    f.write(f"enable_radiation = {rad_new}\n")
                    if k_pairs:
                        f.write("\n[k_table]\n")
                        for t, v in zip(*k_pairs):
                            f.write(f"{t},{v}\n")
                    if cp_pairs:
                        f.write("\n[cp_table]\n")
                        for t, v in zip(*cp_pairs):
                            f.write(f"{t},{v}\n")
                st.success(f"✅ Saved '{safe_name}' to materials_db/")
                st.session_state["show_editor"] = False
                st.rerun()

            if close_btn:
                st.session_state["show_editor"] = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Calculation method & simulation ---
with st.sidebar.expander("🧮 Calculation method & simulation", expanded=False):
    SEG_LEN_mm = st.number_input("Lump length (mm)", min_value=0.1, value=10.0, step=1.0)
    BED_FRAC = st.slider("Bed contact fraction (layer 0)", 0.0, 1.0, 1.0, 0.05)
    CONTACT_FR = st.slider("Layer–layer contact fraction", 0.0, 1.0, 1.0, 0.05)
    DT = st.number_input("dt (s)", min_value=0.001, value=0.050, step=0.001, format="%.3f")

    try:
        tseg = None
        cache = getattr(st.session_state, "cache", None)
        if isinstance(cache, dict):
            t_raw = cache.get("t", None)
            if t_raw is not None:
                t_arr = np.asarray(t_raw, dtype=float)
                if t_arr.size > 1:
                    dt_segs = np.diff(t_arr)
                    dt_segs = dt_segs[dt_segs > 1e-6]
                    if dt_segs.size:
                        tseg = float(np.median(dt_segs))
        if tseg is None:
            v_user = float(st.session_state.get("HINT_SPEED_MM_S", 20.0))
            tseg = SEG_LEN_mm / max(1e-6, v_user)
            src = f"(using your speed {v_user:.0f} mm/s)"
        else:
            src = "(from loaded G-code)"
        lo, hi = 0.10 * tseg, 0.50 * tseg
        st.caption(f"For the current lump size, one-segment time ≈ {tseg:.3f} s {src}. Try Δt in [{lo:.3f}, {hi:.3f}] s.")
    except Exception:
        pass

    COOLDOWN = st.number_input("Post-cooldown (s)", min_value=0.0, value=0.0, step=5.0)
    SNAP_INT = st.number_input("Snapshot every (s)", min_value=0.1, value=5.0, step=0.5)
    LINK_MAX_F = st.number_input("Horizontal link × SEG_LEN", min_value=1.0, value=1.10, step=0.05)
    V_RAD_MM = st.text_input("Vertical search radius (mm, blank = 0.6×W)", "")
    MARKER_SIZE = st.slider("Marker size (GIF)", 5, 60, 18)

    use_mg = st.checkbox("Enable microgrid", value=bool(st.session_state.get("MICROGRID_ENABLE", False)))
    mg_nz = st.number_input("Nz", 1, 50, int(st.session_state.get("MG_NZ", 5)), 1, disabled=not use_mg)
    mg_nx = st.number_input("Nx", 1, 100, int(st.session_state.get("MG_NX", 20)), 1, disabled=not use_mg)
    st.session_state["MICROGRID_ENABLE"] = bool(use_mg)
    st.session_state["MG_NZ"] = int(mg_nz)
    st.session_state["MG_NX"] = int(mg_nx)

# ===== Actions =====
st.sidebar.markdown("### Actions")
run_side = st.sidebar.button("Run (Calculate)")
replay_side = st.sidebar.button("Replay")
clear_side = st.sidebar.button("Clear outputs")

if "cache" in st.session_state and "g_opt" in st.session_state.cache:
    st.sidebar.download_button(
        "⬇️ Download optimized G-code",
        data=st.session_state.cache["g_opt"],
        file_name="optimized.gcode", mime="text/plain"
    )

DIRS, GIFS = build_output_dirs()
S = st.session_state
init_state(S)
ui = UI(DIRS, GIFS)
logs_slot = st.empty()
pdf_slot = st.empty()

c1, c2, c3 = st.columns(3)
run_top = c1.button("Run (Calculate)")
replay_top = c2.button("Replay")
clear_top = c3.button("Clear outputs")

run_clicked = run_top or run_side
replay_clicked = replay_top or replay_side
clear_clicked = clear_top or clear_side

def _purge_all():
    from app_utils import purge_frames
    for dir_map in (DIRS["base"], DIRS["opt"]):
        for d in dir_map.values():
            purge_frames(d)
    for p in [*GIFS["base"].values(), *GIFS["opt"].values()]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    S.cache.clear()
    S.has_base = False
    ui.clear_all()
    logs_slot.empty()
    pdf_slot.empty()

if clear_clicked:
    _purge_all()
    st.success("Cleared.")

if replay_clicked:
    S.replay_tick += 1
    ui.show_everything_from_cache(S, last_frame_path)

if run_clicked:
    if not gcode_file:
        st.error("Upload a G-code first.")
    else:
        with st.status("Running base simulation...", expanded=True) as status:
            try:
                run_base(
                    gbytes=gcode_file.read(),
                    SEG_LEN_mm=SEG_LEN_mm, SEG_WIDTH_mm=SEG_WIDTH_mm, SEG_HEIGHT_mm=SEG_HEIGHT_mm,
                    RHO=RHO, CP=CP, K=K,
                    T_NOZZLE=T_NOZZLE, T_BED=T_BED, T_INF=T_INF, H_COEF=H_COEF,
                    BED_FRAC=BED_FRAC, CONTACT_FR=CONTACT_FR,
                    DT=DT, COOLDOWN=COOLDOWN, SNAP_INT=SNAP_INT,
                    LINK_MAX_F=LINK_MAX_F, V_RAD_MM=V_RAD_MM,
                    MARKER_SIZE=MARKER_SIZE,
                    EMISSIVITY=EMISSIVITY, ENABLE_RADIATION=ENABLE_RADIATION,
                    S=S, ui=ui, GIFS=GIFS, DIRS=DIRS,
                    logs_slot=logs_slot, pdf_slot=pdf_slot
                )
                status.update(label="Base simulation completed")
            except Exception as e:
                st.error(f"Run failed: {e}")
                S.has_base = False

st.subheader("Optimization (per-layer time to hit Substrate temperature)")
with st.form("opt_form"):
    d1, d2, d3 = st.columns([1.5, 1, 1])
    opt_target = d1.number_input("Target Substrate temperature (°C)", value=120.0, step=5.0)
    opt_iters = d2.number_input("Max bisection steps / layer", value=6, min_value=1, max_value=12)
    opt_tol = d3.number_input("Tolerance (°C)", value=2.0, min_value=0.1, step=0.5)

    e1, e2 = st.columns(2)
    opt_smin = e1.number_input("Min time ×", value=0.5, step=0.1)
    opt_smax = e2.number_input("Max time ×", value=2.5, step=0.1)

    submitted_opt = st.form_submit_button("Optimize to target", disabled=not S.get("has_base", False))

if submitted_opt and S.has_base:
    with st.status("Optimizing per-layer time...", expanded=True) as status:
        try:
            optimize_to_target(
                opt_target=opt_target, opt_iters=int(opt_iters), opt_tol=float(opt_tol),
                opt_smin=float(opt_smin), opt_smax=float(opt_smax),
                SEG_LEN_mm=SEG_LEN_mm, SEG_WIDTH_mm=SEG_WIDTH_mm, SEG_HEIGHT_mm=SEG_HEIGHT_mm,
                RHO=RHO, CP=CP, K=K,
                T_NOZZLE=T_NOZZLE, T_BED=T_BED, T_INF=T_INF, H_COEF=H_COEF,
                BED_FRAC=BED_FRAC, CONTACT_FR=CONTACT_FR,
                DT=DT, COOLDOWN=COOLDOWN, SNAP_INT=SNAP_INT,
                LINK_MAX_F=LINK_MAX_F, V_RAD_MM=V_RAD_MM,
                MARKER_SIZE=MARKER_SIZE,
                EMISSIVITY=EMISSIVITY, ENABLE_RADIATION=ENABLE_RADIATION,
                S=S, ui=ui, GIFS=GIFS, DIRS=DIRS, logs_slot=logs_slot
            )
            status.update(label="Optimization complete")
        except Exception as e:
            st.error(f"Optimization failed: {e}")

ui.export_section(S)
if S.has_base and not (run_clicked or replay_clicked or clear_clicked or submitted_opt):
    ui.show_everything_from_cache(S, last_frame_path)
