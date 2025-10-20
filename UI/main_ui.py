import streamlit as st
from handlers import optimize_to_target
from app_utils import has_opt_gifs


def show_optimization_section(S, ui, inputs, DIRS, GIFS, last_frame_path):
    """
    Unified results + optional optimization section.
    The optimization form only appears after a base simulation is completed.
    """

    # --- No simulation yet ---
    if not getattr(S, "has_base", False):
        st.info("⚙️ Run the simulation to view results here.")
        return

    st.markdown(
        """
        <style>
            div.streamlit-expanderHeader p {
                font-size: 1.2rem;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Simulation Results ---
    with st.expander("Simulation Results", expanded=True):
        st.subheader("Printed Path & Layer Metrics")
        ui.draw_base_charts(S)
        ui.show_gifs("base", S, last_frame_path)

    st.markdown("---")

    # --- Optimization controls ---
    submitted_opt = False
    with st.expander("Optimization Setup", expanded=True):
        st.caption("Adjust per-layer speed to reach a target substrate temperature.")

        with st.form("opt_form"):
            d1, d2, d3 = st.columns([1.5, 1, 1])
            opt_target = d1.number_input(
                "Target substrate temperature (°C)", value=120.0, step=5.0
            )
            opt_iters = d2.number_input(
                "Max bisection steps per layer", value=6, min_value=1, max_value=12
            )
            opt_tol = d3.number_input(
                "Tolerance (°C)", value=2.0, min_value=0.1, step=0.5
            )

            e1, e2 = st.columns(2)
            opt_smin = e1.number_input("Min time ×", value=0.5, step=0.1)
            opt_smax = e2.number_input("Max time ×", value=2.5, step=0.1)

            submitted_opt = st.form_submit_button("Run Optimization")

    if submitted_opt and S.has_base:
        with st.status("Optimizing per-layer time...", expanded=True) as status:
            try:
                optimize_to_target(
                    opt_target=opt_target,
                    opt_iters=int(opt_iters),
                    opt_tol=float(opt_tol),
                    opt_smin=float(opt_smin),
                    opt_smax=float(opt_smax),
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
                    logs_slot=st.empty(),
                )
                status.update(label="✅ Optimization complete")
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")

    has_opt_results = "pp_opt" in S.cache and "pm_opt" in S.cache
    with st.expander("Optimized Results", expanded=has_opt_results):
        if has_opt_results:
            ui.draw_opt_overlay_if_available(S)
            if has_opt_gifs(GIFS):
                st.markdown("#### Thermal Profile History (Optimized)")
                ui.show_gifs("opt", S, last_frame_path)
        else:
            ui.clear_opt_overlay_slots()
            st.info("Run the optimization to see results here.")

    ui.export_section(S)
