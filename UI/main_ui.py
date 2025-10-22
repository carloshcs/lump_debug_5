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
        ui.clear_all()
        st.info("⚙️ Run the simulation to view results here.")
        return

    if S.get("inputs_dirty", False):
        ui.clear_all()
        st.warning("Inputs changed since the last run. Re-run the simulation to refresh the results.")
        return

    st.markdown(
        """
        <style>
            div.streamlit-expanderHeader p {
                font-size: 1.6rem;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Simulation Results ---
    with st.expander("Simulation Results", expanded=True):
        charts_tab, views_tab = st.tabs(["Charts", "Views"])
        with charts_tab:
            st.subheader("Printed Path & Layer Metrics")
            ui.draw_base_charts(S)
        with views_tab:
            st.subheader("Simulation Views")
            ui.show_gifs("base", S, last_frame_path)

    st.markdown("---")

    # --- Optimization controls ---
    submitted_opt = False
    with st.expander("Optimization Setup", expanded=True):
        st.caption("Adjust per-layer speed to reach a target substrate temperature.")

        with st.form("opt_form"):
            col_left, col_right = st.columns(2)
            with col_left:
                opt_target = st.number_input(
                    "Target substrate temperature (°C)",
                    value=120.0,
                    step=5.0,
                    help="Desired substrate temperature measured at the start of the next layer.",
                )
                opt_iters = st.number_input(
                    "Max bisection steps per layer",
                    value=6,
                    min_value=1,
                    max_value=12,
                    help="Higher values improve accuracy but lengthen the optimization.",
                )
                opt_smin = st.number_input(
                    "Min time ×",
                    value=0.5,
                    step=0.1,
                    help="Fastest allowable per-layer time multiplier.",
                )
            with col_right:
                opt_tol = st.number_input(
                    "Tolerance (°C)",
                    value=2.0,
                    min_value=0.1,
                    step=0.5,
                    help="Stop iterating when the error drops below this value.",
                )
                opt_smax = st.number_input(
                    "Max time ×",
                    value=2.5,
                    step=0.1,
                    help="Slowest allowable per-layer time multiplier.",
                )

            submitted_opt = st.form_submit_button("Run Optimization")

    if submitted_opt and getattr(S, "has_base", False):
        with st.status("Optimizing per-layer time...", expanded=True) as status:
            log_holder = st.container()
            logs_slot = log_holder.empty()
            progress_lines = []

            def _push_progress(msg: str):
                progress_lines.append(str(msg))
                logs_slot.code("\n".join(progress_lines))

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
                    USE_TABULAR=inputs["USE_TABULAR"],
                    K_TABLE=inputs.get("K_TABLE"),
                    CP_TABLE=inputs.get("CP_TABLE"),
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
                    progress_cb=_push_progress,
                )
                status.update(label="✅ Optimization complete")
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")

    has_opt_results = "pp_opt" in S.cache and "pm_opt" in S.cache
    if has_opt_results:
        with st.expander("Optimization Results", expanded=True):
            overlay_tab, opt_views_tab = st.tabs(["Comparison Charts", "Optimized Views"])
            with overlay_tab:
                ui.draw_opt_overlay_if_available(S)
            with opt_views_tab:
                if has_opt_gifs(GIFS):
                    ui.show_gifs("opt", S, last_frame_path)
                else:
                    st.info("Run the optimization to generate optimized thermal views.")
    else:
        ui.clear_opt_overlay_slots()

    ui.export_section(S)
