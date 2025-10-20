# viz.py
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import plotly.graph_objects as go
import plotly.express as px

# ── Plotly path (classic layout)
def plot_path_interactive(x, y, zc, bounds, title):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x, y=y, z=zc,
            mode="lines",                       # lines only
            line=dict(width=3, color="blue"),   # solid blue line
            hoverinfo="skip"                    # optional: cleaner hover
        )]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax]),
            zaxis=dict(range=[zmin, zmax]),
            aspectmode="data"
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_layer_times(layer_time, title):
    Ls = [L for (L, _) in layer_time]
    Ts = [t for (_, t) in layer_time]

    fig = px.line(
        x=Ls,
        y=Ts,
        markers=True,
        labels={"x": "Layer number", "y": "Layer time (s)"},
        title=title
    )

    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".1f")

    # --- Add a fixed or padded y-axis range ---
    if len(Ts) > 0:
        t_min, t_max = min(Ts), max(Ts)
        if abs(t_max - t_min) < 1e-2:  # almost constant
            # show a flat line with small buffer
            y_pad = 0.05  # 0.05 s margin around the mean
            y_center = 0.5 * (t_max + t_min)
            fig.update_yaxes(range=[y_center - y_pad, y_center + y_pad])
        else:
            # otherwise add 5% padding to normal ranges
            pad = 0.05 * (t_max - t_min)
            fig.update_yaxes(range=[t_min - pad, t_max + pad])

    return fig



def plot_layer_times_overlay(base_layer_time, opt_layer_time):
    import plotly.graph_objects as go

    fig = go.Figure()

    def add_trace(name, pairs, dash=None):
        if not pairs:
            return
        xs = [L for (L, _) in pairs]
        ys = [t for (_, t) in pairs]
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=name,
            line=(dict(dash=dash) if dash else None)
        )

    add_trace("Base", base_layer_time)
    add_trace("Optimized", opt_layer_time, dash="dash")

    fig.update_layout(
        title="Layer time per layer (base vs optimized)",
        xaxis_title="Layer number",
        yaxis_title="Layer time (s)"
    )
    fig.update_xaxes(dtick=1)

    return fig



def plot_substracte(pairs_point, pairs_mean, title):
    Lp = [L for (L,_) in pairs_point]; Tp = [T for (_,T) in pairs_point]
    Lm = [L for (L,_) in pairs_mean];  Tm = [T for (_,T) in pairs_mean]
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=Lp, y=Tp, mode="lines+markers", name="Point @ next-layer start")
    fig.add_scatter(x=Lm, y=Tm, mode="lines+markers", name="Layer mean @ next-layer start", line=dict(dash="dash"))
    fig.update_layout(title=title, xaxis_title="Layer number", yaxis_title="Temperature (°C)")
    fig.update_xaxes(dtick=1)
    return fig

def plot_substracte_overlay(base_point, base_mean, opt_point, opt_mean):
    import plotly.graph_objects as go
    fig = go.Figure()
    def add(name, pairs, dash=None):
        if not pairs: return
        xs = [L for (L,_) in pairs]; ys = [T for (_,T) in pairs]
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=name,
                        line=(dict(dash=dash) if dash else None))
    add("Base point", base_point)
    add("Base mean",  base_mean,  dash="dash")
    add("Opt point",  opt_point)
    add("Opt mean",   opt_mean,   dash="dash")
    fig.update_layout(title="Layer Substracte Temperature (base vs optimized)",
                      xaxis_title="Layer number", yaxis_title="Temperature (°C)")
    fig.update_xaxes(dtick=1)
    return fig

# ── Matplotlib GIF frames (same layout/size as your original; now a line colored by T)
def _clean_axes(ax, view):
    if view in ("top","front"):
        for lab in (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel):
            lab("")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        try:
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")
        except Exception:
            pass
        ax.grid(False)

def render_frames_from_states(states, bounds, out_dir, view,
                              vmin=20.0, vmax=300.0, marker_size=16, dpi=240,
                              clean=True,
                              # compat args (ignored but accepted by callers):
                              draw_style="line", cmap="viridis", linewidth=2.0):
    """
    states: list of snapshots, each like {"pos": Nx3, "T": N, "time": float}
    bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    view:   "iso" | "top" | "front"
    Renders a temperature-colored LINE (no dots) following the toolpath.
    """
    os.makedirs(out_dir, exist_ok=True)
    xmin,xmax,ymin,ymax,zmin,zmax = bounds
    Xbed, Ybed = np.meshgrid([xmin, xmax], [ymin, ymax]); Zbed = np.zeros_like(Xbed)

    # color mapping
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

    for idx, s in enumerate(states):
        P = s["pos"]; T = s["T"]; t_now = s["time"]
        if P is None or T is None or np.size(P) == 0 or np.size(T) == 0:
            continue

        P = np.asarray(P); T = np.asarray(T).reshape(-1)
        n = min(P.shape[0], T.shape[0])
        if n < 2:
            continue
        P = P[:n, :3]; T = T[:n]

        # segments between consecutive points + per-segment color
        segs = np.stack([P[:-1, :3], P[1:, :3]], axis=1)  # (n-1,2,3)
        cvals = 0.5 * (T[:-1] + T[1:])

        # figure/axes: EXACT same size as before
        fig = plt.figure(figsize=(6.0, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        
        if view == "iso":
            ax.grid(False)
            # also kill the faint pane gridlines in 3D (some backends draw them)
            try:
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis._axinfo["grid"]["linewidth"] = 0.0
            except Exception:
                pass
        
        # bed plane (light gray)
        ax.plot_surface(Xbed, Ybed, Zbed, color="lightgray", alpha=0.35, edgecolor="none")

        # temperature-colored line path (replaces scatter)
        lc = Line3DCollection(segs, cmap=cmap_obj, norm=norm, linewidths=2.0)
        lc.set_array(cvals)
        ax.add_collection3d(lc)

        # colorbar (consistent scale)
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.08)
        cbar.set_label("Temperature (°C)")

        # limits + labels + views (unchanged so panel sizes match your originals)
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        if view == "iso":   ax.view_init(elev=22, azim=-60)
        if view == "top":   ax.view_init(elev=90, azim=-90)
        if view == "front": ax.view_init(elev=0,  azim=0)

        if clean: _clean_axes(ax, view)
        ax.set_title(f"t = {t_now:.1f} s  |  lumps = {n}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frame_{idx:04d}.png"), dpi=dpi )
        plt.close(fig)

def build_gif_from_frames(frames_dir, gif_path, fps=2):
    import imageio.v2 as imageio
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not files: return
    imgs = [imageio.imread(f) for f in files]
    dur = 1.0/float(fps) if fps>0 else 0.5
    imageio.mimsave(gif_path, imgs, duration=dur)

def last_frame_path(frames_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    return files[-1] if files else ""

# ── Optional PDF (safe if reportlab not installed)
def make_pdf_report(pdf_path: str, image_paths: list, title: str = "Thermal report", notes: str = ""):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception:
        # reportlab not available → do nothing
        return None

    W, H = A4
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setTitle(title)

    y = H - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 20
    c.setFont("Helvetica", 9)
    if notes:
        txt = c.beginText(40, y)
        for line in notes.splitlines():
            txt.textLine(line[:120])
            y -= 11
        c.drawText(txt)
        y -= 6

    for p in image_paths:
        if not p: continue
        try:
            img = ImageReader(p)
            iw, ih = img.getSize()
            maxw, maxh = W-80, 260
            scale = min(maxw/iw, maxh/ih)
            w, h = iw*scale, ih*scale
            if y - h < 40:
                c.showPage(); y = H - 40
            c.drawImage(img, 40, y-h, width=w, height=h, preserveAspectRatio=True)
            y -= (h + 20)
        except Exception:
            continue

    c.showPage()
    c.save()
    return pdf_path
