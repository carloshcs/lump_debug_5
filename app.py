import os

# ✅ Disable watchdog file watcher (fixes RuntimeError: dictionary changed size during iteration)
os.environ["WATCHDOG_DISABLE_FILE_WATCH"] = "true"

import streamlit as st
from app_utils import build_output_dirs, init_state
from UI.ui import UI
from viz import last_frame_path
from UI.sidebar_inputs import get_inputs
from UI.material_editor import show_material_editor
from UI.actions import handle_actions
from UI.main_ui import show_optimization_section


# --- Initialization ---
S = st.session_state
if "mode" not in S:
    S.mode = "home"  # "home", "editor", or "simulation"
if "simulation_started" not in S:
    S.simulation_started = False
if "show_editor" not in S:
    S.show_editor = False

DIRS, GIFS = build_output_dirs()
init_state(S)
ui = UI(DIRS, GIFS)


def set_layout(layout="wide"):
    st.set_page_config(page_title="Lumped Thermal Simulator", layout=layout)


# ================================
# MAIN VIEW CONTROLLER
# ================================
set_layout("wide")

# --- Always show sidebar ---
with st.sidebar:
    inputs = get_inputs(S, key_prefix="main")
    handle_actions(S, ui, inputs, DIRS, GIFS)

# --- Main area ---
if S.mode == "home":
    S.show_editor = False  # Hide editor when returning home

    st.title("Lumped Thermal Simulator")
    st.caption("Mode: Home")

    # 🧩 Quick Tutorial (only shown on home)
    st.markdown(
        """
        <div style="background-color:#fafafa;padding:20px;border-radius:10px;margin-bottom:20px;">
        <h4 style="margin-top:0;">🧩 Quick Tutorial</h4>
        <ol>
          <li>Upload G-code file</li>
          <li>Set printing parameters</li>
          <li>Adjust material and simulation settings</li>
          <li>Click <b>Run Simulation</b></li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif S.mode == "editor":
    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.title("Material Editor")
        st.caption("Mode: Editor")
    with action_col:
        if st.button("🏠 Home", use_container_width=True):
            S.mode = "home"
            S.show_editor = False
            st.rerun()

    # Display the material editor
    show_material_editor(inputs["MATERIALS_DIR"], inputs["MATERIALS"], inputs["parse_table_text"])

    # If editor closed, return automatically to home
    if not S.get("show_editor", True):
        S.mode = "home"
        st.rerun()

    st.markdown("---")
    if st.button("🏠 Back to Home"):
        S.mode = "home"
        S.show_editor = False
        st.rerun()

elif S.mode == "simulation":
    # Ensure material editor is hidden in results page
    S.show_editor = False

    # --- 🧠 Results Page ---
    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.title("Thermal Simulation Results")
        st.caption("Mode: Simulation")
    with action_col:
        if st.button("🏠 Home", use_container_width=True, key="top_home"):
            S.mode = "home"
            S.simulation_started = False
            S.show_editor = False
            st.rerun()

    show_optimization_section(S, ui, inputs, DIRS, GIFS, last_frame_path)

    st.markdown("---")
    if st.button("🏠 Return to Home"):
        S.mode = "home"
        S.simulation_started = False
        S.show_editor = False
        st.rerun()

else:
    st.stop()
