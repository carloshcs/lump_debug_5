import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def show_material_editor(MATERIALS_DIR, MATERIALS, parse_table_text):
    """Material Editor with centered table/chart titles."""

    if not st.session_state.get("show_editor", False):
        return

    # --- Layout ---
    st.markdown(
        """
        <style>
        .main > div {
            max-width: 900px;
            margin: 0 auto;
        }
        .centered-title {
            text-align: center;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 10px;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Mode selection ---
    mode = st.radio(
        "Select mode",
        ["✏️ Edit existing material", "➕ Add new material"],
        horizontal=True,
    )
    add_mode = mode == "➕ Add new material"

    # --- Material list or new entry ---
    materials = sorted(MATERIALS.keys())
    if add_mode:
        selected_material = None
        existing = {}
        name_new = st.text_input("Material name", value="", placeholder="Enter new material name")
    else:
        if not materials:
            st.warning("No materials found. Please add a new one first.")
            return
        selected_material = st.selectbox("Select material to edit", materials, key="selected_material")
        existing = MATERIALS.get(selected_material, {})
        name_new = existing.get("name", selected_material)

    # --- Helpers ---
    def pairs_to_df(pairs, default_rows=2):
        if not pairs:
            return pd.DataFrame({"T": ["" for _ in range(default_rows)],
                                 "Value": ["" for _ in range(default_rows)]})
        T, V = pairs
        df = pd.DataFrame({"T": T, "Value": V})
        while len(df) < default_rows:
            df.loc[len(df)] = ["", ""]
        return df

    def df_to_pairs(df):
        df = df.dropna().astype(str)
        df = df[(df["T"] != "") & (df["Value"] != "")]
        return (df["T"].tolist(), df["Value"].tolist()) if len(df) > 0 else None

    # --- Defaults ---
    rho_default = float(existing.get("rho", 1200.0))
    k_default = float(existing.get("k", 0.25))
    cp_default = float(existing.get("cp", 1300.0))
    emi_default = float(existing.get("emissivity", 0.85))
    k_df_default = pairs_to_df(existing.get("k_table"))
    cp_df_default = pairs_to_df(existing.get("cp_table"))

    # --- Property Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        rho_new = st.number_input("Density ρ (kg/m³)", value=rho_default, step=10.0)
        cp_new = st.number_input("Specific heat cₚ (J/kg·K)", value=cp_default, step=10.0)
    with col2:
        k_new = st.number_input("Conductivity k (W/m·K)", value=k_default, step=0.01, format="%.3f")
        emi_new = st.number_input("Emissivity ε (–)", 0.0, 1.0, emi_default, 0.05)

    # --- Tables + Graphs ---
    st.markdown("### 📈 Temperature-dependent properties")

    # --- Thermal conductivity ---
    colk1, colk2 = st.columns([1, 1])
    with colk1:
        st.markdown('<div class="centered-title">Thermal conductivity coefficient (k)</div>', unsafe_allow_html=True)
        k_df = st.data_editor(
            k_df_default,
            use_container_width=True,
            column_config={"T": "T (°C)", "Value": "k (W/m·K)"},
            num_rows="dynamic",
            key="k_table_edit",
        )
    with colk2:
        if not k_df.empty and k_df["T"].iloc[0] != "":
            try:
                fig_k = go.Figure()
                fig_k.add_trace(go.Scatter(
                    x=pd.to_numeric(k_df["T"], errors="coerce"),
                    y=pd.to_numeric(k_df["Value"], errors="coerce"),
                    mode="lines+markers",
                    name="k(T)"
                ))
                fig_k.update_layout(
                    title={
                        "text": f"Thermal conductivity of {name_new}",
                        "x": 0.5,  # centered
                        "xanchor": "center"
                    },
                    xaxis_title="T (°C)",
                    yaxis_title="k (W/m·K)",
                )
                st.plotly_chart(
                    fig_k,
                    use_container_width=True,
                    key="mat_k_plot"
                )
            except Exception:
                st.warning("⚠️ Invalid data in k(T) table.")

    # --- Specific heat ---
    colcp1, colcp2 = st.columns([1, 1])
    with colcp1:
        st.markdown('<div class="centered-title">Specific heat coefficient (cp)</div>', unsafe_allow_html=True)
        cp_df = st.data_editor(
            cp_df_default,
            use_container_width=True,
            column_config={"T": "T (°C)", "Value": "cp (J/kg·K)"},
            num_rows="dynamic",
            key="cp_table_edit",
        )
    with colcp2:
        if not cp_df.empty and cp_df["T"].iloc[0] != "":
            try:
                fig_cp = go.Figure()
                fig_cp.add_trace(go.Scatter(
                    x=pd.to_numeric(cp_df["T"], errors="coerce"),
                    y=pd.to_numeric(cp_df["Value"], errors="coerce"),
                    mode="lines+markers",
                    name="cp(T)"
                ))
                fig_cp.update_layout(
                    title={
                        "text": f"Specific heat of {name_new}",
                        "x": 0.5,  # centered
                        "xanchor": "center"
                    },
                    xaxis_title="T (°C)",
                    yaxis_title="cp (J/kg·K)",
                )
                st.plotly_chart(
                    fig_cp,
                    use_container_width=True,
                    key="mat_cp_plot"
                )
            except Exception:
                st.warning("⚠️ Invalid data in cp(T) table.")

    # --- Buttons ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save"):
            safe_name = name_new.strip().replace(" ", "_") or "Unnamed"
            k_pairs = df_to_pairs(k_df)
            cp_pairs = df_to_pairs(cp_df)
            path = os.path.join(MATERIALS_DIR, f"{safe_name}.txt")

            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"name = {safe_name}\n")
                    f.write(f"rho = {rho_new}\n")
                    f.write(f"k = {k_new}\n")
                    f.write(f"cp = {cp_new}\n")
                    f.write(f"emissivity = {emi_new}\n")
                    if k_pairs:
                        f.write("\n[k_table]\n")
                        for t, v in zip(*k_pairs):
                            f.write(f"{t},{v}\n")
                    if cp_pairs:
                        f.write("\n[cp_table]\n")
                        for t, v in zip(*cp_pairs):
                            f.write(f"{t},{v}\n")

                st.success(f"✅ Saved '{safe_name}' successfully.")
                st.session_state["show_editor"] = False
                st.session_state["mode"] = "home"
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save material: {e}")

    with c2:
        if st.button("❌ Close"):
            st.session_state["show_editor"] = False
            st.session_state["mode"] = "home"
            st.rerun()
