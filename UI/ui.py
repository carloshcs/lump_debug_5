# -*- coding: utf-8 -*-
"""
UI components for the Lumped Thermal Simulator
"""

import time
from typing import Dict
import streamlit as st
from viz import (
    plot_path_interactive,
    plot_layer_times,
    plot_layer_times_overlay,
    plot_substracte,
    plot_substracte_overlay,
)

from app_utils import load_view_bytes, has_opt_gifs


class UI:
    def __init__(self, dirs: Dict, gifs: Dict):
        self.DIRS = dirs
        self.GIFS = gifs

        # Containers are instantiated lazily so the results appear exactly
        # where the caller places them in the Streamlit script.
        self.base_chart_container = None
        self.base_gif_container = None
        self.overlay_container = None
        self.opt_gif_container = None

        # Slots initialised on first use
        self.path_slot = None
        self.times_slot = None
        self.sub_header_slot = None
        self.sub_base_slot = None

        self.iso_base_slot = None
        self.top_base_slot = None
        self.front_base_slot = None

        self.sub_overlay_slot = None
        self.opt_times_overlay_slot = None
        self.opt_sub_header_slot = None

        self.iso_opt_slot = None
        self.top_opt_slot = None
        self.front_opt_slot = None

    # ----------- helpers -----------
    def _next_chart_key(self, S, name: str) -> str:
        """Return a unique Streamlit component key for charts.

        Streamlit re-creates the widgets on every run and relies on the
        developer-provided ``key`` to distinguish components that would
        otherwise look identical.  Without unique keys, subsequent
        re-renders (for example during optimisation) may trigger
        duplicated component ID errors.  We keep a counter on the state
        object so that each chart rendered through the UI gets a fresh
        key.
        """
        S.chart_tick += 1
        return f"{name}_{S.chart_tick}"

    def _ensure_base_chart_slots(self):
        if self.base_chart_container is None:
            self.base_chart_container = st.container()
            with self.base_chart_container:
                self.path_slot = st.empty()
                self.times_slot = st.empty()
                self.sub_header_slot = st.empty()
                self.sub_base_slot = st.empty()

    def _ensure_base_gif_slots(self):
        if self.base_gif_container is None:
            self.base_gif_container = st.container()
            with self.base_gif_container:
                row_base = st.columns(3)
                self.iso_base_slot = row_base[0].empty()
                self.top_base_slot = row_base[1].empty()
                self.front_base_slot = row_base[2].empty()

    def _ensure_overlay_slot(self):
        if self.overlay_container is None:
            self.overlay_container = st.container()
            with self.overlay_container:
                self.opt_times_overlay_slot = st.empty()
                self.opt_sub_header_slot = st.empty()
                self.sub_overlay_slot = st.empty()

    def _ensure_opt_gif_slots(self):
        if self.opt_gif_container is None:
            self.opt_gif_container = st.container()
            with self.opt_gif_container:
                row_opt = st.columns(3)
                self.iso_opt_slot = row_opt[0].empty()
                self.top_opt_slot = row_opt[1].empty()
                self.front_opt_slot = row_opt[2].empty()

    # ----------- clears -----------
    def clear_base_gif_slots(self):
        for slot_name in ("iso_base_slot", "top_base_slot", "front_base_slot"):
            slot = getattr(self, slot_name, None)
            if slot is not None:
                slot.empty()

    def clear_opt_gif_slots(self):
        for slot_name in ("iso_opt_slot", "top_opt_slot", "front_opt_slot"):
            slot = getattr(self, slot_name, None)
            if slot is not None:
                slot.empty()

    def clear_all(self):
        for slot_name in ("path_slot", "times_slot", "sub_base_slot"):
            slot = getattr(self, slot_name, None)
            if slot is not None:
                slot.empty()
        if self.sub_header_slot is not None:
            self.sub_header_slot.empty()
        self.clear_base_gif_slots()
        if self.sub_overlay_slot is not None:
            self.sub_overlay_slot.empty()
        if self.opt_times_overlay_slot is not None:
            self.opt_times_overlay_slot.empty()
        if self.opt_sub_header_slot is not None:
            self.opt_sub_header_slot.empty()
        self.clear_opt_gif_slots()

    def clear_opt_overlay_slots(self):
        for slot_name in ("opt_times_overlay_slot", "opt_sub_header_slot", "sub_overlay_slot"):
            slot = getattr(self, slot_name, None)
            if slot is not None:
                slot.empty()

    # ----------- draws -----------
    def draw_base_charts(self, S):
        self._ensure_base_chart_slots()

        if "t" in S.cache and "bounds" in S.cache:
            self.path_slot.plotly_chart(
                plot_path_interactive(
                    S.cache["x"], S.cache["y"], S.cache["zc"], S.cache["bounds"],
                    f"Printed path | length={S.cache.get('Ltot', '?')} m | lumps={len(S.cache['t'])}"
                ),
                use_container_width=True,
                key=self._next_chart_key(S, "plot_path"),
            )
        if "info" in S.cache:
            self.times_slot.plotly_chart(
                plot_layer_times(S.cache["info"]["layer_time"], "Layer time per layer (base)"),
                use_container_width=True,
                key=self._next_chart_key(S, "plot_layer_times_base"),
            )
        if "pairs_point" in S.cache and "pairs_mean" in S.cache:
            if self.sub_header_slot is not None:
                self.sub_header_slot.subheader("Thermal Profile History")
            self.sub_base_slot.plotly_chart(
                plot_substracte(S.cache["pairs_point"], S.cache["pairs_mean"], "Layer Substrate Temperature (base)"),
                use_container_width=True,
                key=self._next_chart_key(S, "plot_sub_base"),
            )
        elif self.sub_header_slot is not None:
            self.sub_header_slot.empty()

    def draw_opt_overlay_if_available(self, S):
        if "pp_opt" in S.cache and "pm_opt" in S.cache:
            self._ensure_overlay_slot()
            base_layer_time = S.cache.get("info", {}).get("layer_time", [])
            opt_layer_time = S.cache.get("info_opt", {}).get("layer_time", [])
            if self.opt_times_overlay_slot is not None:
                if base_layer_time or opt_layer_time:
                    self.opt_times_overlay_slot.plotly_chart(
                        plot_layer_times_overlay(base_layer_time, opt_layer_time),
                        use_container_width=True,
                        key=self._next_chart_key(S, "plot_layer_times_overlay"),
                    )
                else:
                    self.opt_times_overlay_slot.empty()
            if self.opt_sub_header_slot is not None:
                self.opt_sub_header_slot.subheader("Thermal Profile History")
            self.sub_overlay_slot.plotly_chart(
                plot_substracte_overlay(
                    S.cache.get("pairs_point", []),
                    S.cache.get("pairs_mean", []),
                    S.cache["pp_opt"],
                    S.cache["pm_opt"]
                ),
                use_container_width=True,
                key=self._next_chart_key(S, f"plot_sub_overlay_{S.overlay_tick}")
            )

    def show_gifs(self, kind: str, S, last_frame_path_fn):
        if kind == "base":
            self._ensure_base_gif_slots()
            slots = (self.iso_base_slot, self.top_base_slot, self.front_base_slot)
        else:
            self._ensure_opt_gif_slots()
            slots = (self.iso_opt_slot, self.top_opt_slot, self.front_opt_slot)

        if kind == "base":
            self.clear_base_gif_slots()
        else:
            self.clear_opt_gif_slots()

        names = ["ISO", "TOP", "FRONT"]
        caps = [("ISO (base)" if kind == "base" else "ISO (opt)"),
                ("TOP (base)" if kind == "base" else "TOP (opt)"),
                ("FRONT (base)" if kind == "base" else "FRONT (opt)")]

        blobs = [None, None, None]
        for _ in range(5):
            for i, nm in enumerate(names):
                if blobs[i] is None:
                    blobs[i] = load_view_bytes(self.DIRS, self.GIFS, kind, nm, S.replay_tick, last_frame_path_fn)
            if all(b is not None for b in blobs):
                break
            time.sleep(0.2)

        tick_suffix = f" • r{S.replay_tick}"
        for s, b, cap in zip(slots, blobs, caps):
            if b is not None:
                s.image(b, caption=cap + tick_suffix, use_container_width=True)

    def show_everything_from_cache(self, S, last_frame_path_fn):
        self.draw_base_charts(S)
        self.show_gifs("base", S, last_frame_path_fn)
        if has_opt_gifs(self.GIFS):
            self.show_gifs("opt", S, last_frame_path_fn)
        self.draw_opt_overlay_if_available(S)

    def _format_substrate_txt(self, pairs_point, pairs_mean, title: str) -> str:
        d_point = {int(L): float(T) for (L, T) in pairs_point}
        d_mean = {int(L): float(T) for (L, T) in pairs_mean}
        layers = sorted(set(d_point.keys()) | set(d_mean.keys()))
        lines = []
        lines.append(f"# Layer substrate temperature (degC) — {title}")
        lines.append("# columns: layer\tT_point_degC\tT_mean_degC")
        for L in layers:
            tp = d_point.get(L, float("nan"))
            tm = d_mean.get(L, float("nan"))
            lines.append(f"{L}\t{tp:.6f}\t{tm:.6f}")
        return "\n".join(lines) + "\n"

    def export_section(self, S):
        st.markdown("### Export")

        if "g_opt" in S.cache:
            st.download_button(
                "⬇️ Download optimized G-code",
                data=S.cache["g_opt"],
                file_name="optimized.gcode",
                mime="text/plain",
                key="dl_opt_gcode_main"
            )

        if "pairs_point" in S.cache and "pairs_mean" in S.cache:
            base_txt = self._format_substrate_txt(
                S.cache["pairs_point"], S.cache["pairs_mean"], "base"
            )
            st.download_button(
                "⬇️ Download base substrate temps (.txt)",
                data=base_txt,
                file_name="substrate_temps_base.txt",
                mime="text/plain",
                key="dl_sub_base_txt"
            )

        if "pp_opt" in S.cache and "pm_opt" in S.cache:
            opt_txt = self._format_substrate_txt(
                S.cache["pp_opt"], S.cache["pm_opt"], "optimized"
            )
            st.download_button(
                "⬇️ Download optimized substrate temps (.txt)",
                data=opt_txt,
                file_name="substrate_temps_optimized.txt",
                mime="text/plain",
                key="dl_sub_opt_txt"
            )
