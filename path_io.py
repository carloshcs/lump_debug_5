# path_io.py — load & resample toolpath

import numpy as np

def load_time_series(path):
    """
    Accepts:
      5 cols: [t, x_mm, y_mm, H_mm, zc_mm]  or
      4 cols: [t, x_mm, y_mm, zc_mm]
    Returns SI units sorted by time: t(s), x,y,zc (m).
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] not in (4,5):
        raise ValueError("Expected 4 or 5 columns in time_series file.")
    if data.shape[1] == 5:
        t = data[:,0]; x = data[:,1]; y = data[:,2]; zc = data[:,4]
    else:
        t = data[:,0]; x = data[:,1]; y = data[:,2]; zc = data[:,3]
    idx = np.argsort(t)
    t,x,y,zc = t[idx], x[idx], y[idx], zc[idx]
    mm = 1e-3
    return t.astype(float), (x*mm).astype(float), (y*mm).astype(float), (zc*mm).astype(float)

def resample_by_length(t, x, y, zc, seg_len):
    """
    Uniform arc-length resampling of the ENTIRE 3D path.
    Prevents extra lumps on curves; keeps constant lump size = seg_len.
    """
    P = np.column_stack([x, y, zc])
    ds = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    Ltot = float(s[-1])
    if Ltot <= 1e-12:
        return t[:1], x[:1], y[:1], zc[:1], Ltot
    s_new = np.arange(0.0, Ltot + 1e-12, seg_len)
    if s_new[-1] < Ltot - 1e-9:
        s_new = np.append(s_new, Ltot)
    T  = np.interp(s_new, s, t)
    X  = np.interp(s_new, s, x)
    Y  = np.interp(s_new, s, y)
    Zc = np.interp(s_new, s, zc)
    return T, X, Y, Zc, Ltot

def compute_bounds(x, y, zc):
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = 0.0, float(max(0.02, zc.max()))
    pad = 0.05 * max(xmax-xmin, ymax-ymin, zmax-zmin, 1e-6)
    return (xmin-pad, xmax+pad, ymin-pad, ymax+pad, zmin, zmax+pad)
