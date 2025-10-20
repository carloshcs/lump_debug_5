# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:11:59 2025

@author: saunders
"""

# app_utils.py
import os
import time
from typing import Dict, Tuple

def mm_to_m(x) -> float:
    return float(x) * 1e-3

def build_output_dirs(base_dir: str = "outputs", opt_dir: str = "outputs_opt"):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
    dirs = {
        "base": dict(
            ISO=f"{base_dir}/frames_iso",
            TOP=f"{base_dir}/frames_top",
            FRONT=f"{base_dir}/frames_front",
        ),
        "opt": dict(
            ISO=f"{opt_dir}/frames_iso",
            TOP=f"{opt_dir}/frames_top",
            FRONT=f"{opt_dir}/frames_front",
        ),
    }
    gifs = {
        "base": dict(
            ISO=f"{base_dir}/printing_iso.gif",
            TOP=f"{base_dir}/printing_top.gif",
            FRONT=f"{base_dir}/printing_front.gif",
        ),
        "opt": dict(
            ISO=f"{opt_dir}/printing_iso_opt.gif",
            TOP=f"{opt_dir}/printing_top_opt.gif",
            FRONT=f"{opt_dir}/printing_front_opt.gif",
        ),
    }
    # ensure frame dirs exist
    for variant in dirs.values():
        for p in variant.values():
            os.makedirs(p, exist_ok=True)
    return dirs, gifs

def purge_frames(frames_dir: str):
    try:
        for name in os.listdir(frames_dir):
            if name.startswith("frame_") and name.lower().endswith(".png"):
                try:
                    os.remove(os.path.join(frames_dir, name))
                except Exception:
                    pass
    except Exception:
        pass

def read_bytes(path: str, retries: int = 6, delay: float = 0.15):
    if not path:
        return None
    for _ in range(retries):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            time.sleep(delay)
    return None

def last_png_any(dir_path: str):
    try:
        pngs = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".png")]
        if not pngs:
            return None
        pngs.sort(key=lambda p: os.path.getmtime(p))
        return pngs[-1]
    except Exception:
        return None

def inject_gif_comment(blob: bytes, tag_text: str) -> bytes:
    try:
        if blob[:6] not in (b"GIF89a", b"GIF87a"):
            return blob
        idx = blob.rfind(b"\x3B")
        if idx == -1:
            return blob
        msg = ("r:" + str(tag_text)).encode("ascii", "ignore")
        if len(msg) > 255:
            msg = msg[:255]
        comment = b"\x21\xFE" + bytes([len(msg)]) + msg + b"\x00"
        return blob[:idx] + comment + blob[idx:]
    except Exception:
        return blob

def has_opt_gifs(gifs: Dict[str, Dict[str, str]]) -> bool:
    try:
        return all(os.path.exists(p) for p in gifs["opt"].values())
    except Exception:
        return False

def load_view_bytes(dirs: Dict, gifs: Dict, kind: str, name: str, replay_tick: int, last_frame_path_fn):
    # Try GIF first
    gif_path = gifs[kind][name]
    b = read_bytes(gif_path)
    if b is not None:
        return inject_gif_comment(b, f"{kind}-{name}-{replay_tick}")
    # Fallback: last rendered PNG
    frames_dir = dirs[kind][name]
    png_path = None
    try:
        png_path = last_frame_path_fn(frames_dir)
    except Exception:
        png_path = None
    if not png_path or not os.path.exists(png_path):
        png_path = last_png_any(frames_dir)
    return read_bytes(png_path) if png_path else None

def init_state(S):
    if "cache" not in S: S.cache = {}
    if "has_base" not in S: S.has_base = False
    if "replay_tick" not in S: S.replay_tick = 0
    if "overlay_tick" not in S: S.overlay_tick = 0
    if "chart_tick" not in S: S.chart_tick = 0
