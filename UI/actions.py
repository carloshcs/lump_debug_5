import os
import streamlit as st
from handlers import run_base
from app_utils import purge_frames
from viz import last_frame_path


def handle_actions(S, ui, inputs, DIRS, GIFS):
    st.sidebar.markdown("### Actions")

    # --- Navigation ---
    if st.sidebar.button("🏠 Back to Home"):
        S.mode = "home"
        S.simulation_started = False
        S.show_editor = False
        st.rerun()

    # --- Buttons ---
    run_side = st.sidebar.button("▶️ Run Simulation")
    see_results = st.sidebar.button("📊 See Results")
    clear_side = st.sidebar.button("🧹 Clear Outputs")

    st.sidebar.markdown("---")

    # --- Material editor shortcut ---
    if st.sidebar.button("🧪 Add/Edit Material"):
        S.mode = "editor"
        S.show_editor = True
        st.rerun()

    logs_slot = st.empty()
    pdf_slot = st.empty()

    # --- Helper: purge all outputs ---
    def purge_all():
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
        S.last_sim_signature = None
        S.current_signature = None
        S.inputs_dirty = False
        ui.clear_all()
        logs_slot.empty()
        pdf_slot.empty()

    # --- CLEAR ---
    if clear_side:
        purge_all()
        S.mode = "home"
        S.show_editor = False
        S.simulation_started = False
        st.sidebar.success("Cleared all outputs.")

    # --- SEE RESULTS ---
    if see_results:
        if getattr(S, "has_base", False):
            S.mode = "simulation"
            S.show_editor = False
            S.simulation_started = True
        else:
            st.sidebar.warning("There is no result, please run the simulation first.")

    # --- RUN SIMULATION ---
    if run_side:
        gcode_file = inputs["gcode_file"]
        if not gcode_file:
            st.error("Please upload a G-code file first.")
        else:
            with st.status("Running base simulation...", expanded=True) as status:
                try:
                    try:
                        gcode_file.seek(0)
                    except Exception:
                        pass
                    run_base(
                        gbytes=gcode_file.read(),
                        SEG_LEN_mm=inputs["SEG_LEN_mm"],
                        SEG_WIDTH_mm=inputs["SEG_WIDTH_mm"],
                        SEG_HEIGHT_mm=inputs["SEG_HEIGHT_mm"],
                        RHO=inputs["RHO"],
                        CP=inputs["CP"],
                        K=inputs["K"],
                        T_NOZZLE=inputs["T_NOZZLE"],
                        T_BED=inputs["T_BED"],
                        T_INF=inputs["T_INF"],
                        H_COEF=inputs["H_COEF"],
                        BED_FRAC=inputs["BED_FRAC"],
                        CONTACT_FR=inputs["CONTACT_FR"],
                        DT=inputs["DT"],
                        COOLDOWN=inputs["COOLDOWN"],
                        SNAP_INT=inputs["SNAP_INT"],
                        LINK_MAX_F=inputs["LINK_MAX_F"],
                        V_RAD_MM=inputs["V_RAD_MM"],
                        MARKER_SIZE=inputs["MARKER_SIZE"],
                        EMISSIVITY=inputs["EMISSIVITY"],
                        ENABLE_RADIATION=inputs["ENABLE_RADIATION"],
                        S=S,
                        ui=ui,
                        GIFS=GIFS,
                        DIRS=DIRS,
                        logs_slot=logs_slot,
                        pdf_slot=pdf_slot,
                    )
                    status.update(label="✅ Simulation completed")
                    S.simulation_started = True
                    S.mode = "simulation"
                    S.show_editor = False
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    S.has_base = False
