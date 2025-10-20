# model.py — path-based lumped thermal network (area-weighted, many-to-many vertical contact)

import numpy as np

class GrowingLumpModel:
    def __init__(self, seg_len, seg_width, seg_height,
                 rho, cp, k,
                 T_bed, T_amb, h,
                 dt, post_cooldown,
                 link_max, vert_radius=None,
                 T_nozzle=300.0,
                 bed_contact_frac=1.0,      # scales bed-contact AREA (0..1)
                 contact_area_frac=1.0,     # scales layer-layer contact AREA (0..1)
                 microgrid_enable=False, nx=20, nz=10,
                 # ---- Convection per-face multipliers (no UI by default) ----
                 h_top_mul: float = 0.0,    # default: top convection OFF (matches your Abaqus setup)
                 h_side_mul: float = 1.0,   # default: sides at 100% of H_COEF
                 h_bottom_mul: float = 1.0, # default: keep bed coupling as-is (first layer)
                 # ---- Radiation controls (independent from convection) ----
                 emissivity: float = 0.8,   # material emissivity (0..1)
                 enable_radiation: bool = False,  # toggle radiation losses
                 rad_top_mul: float = 0.0,        # default: NO top radiation
                 rad_side_mul: float = 1.0,       # default: radiation on sides/ends
                 rad_bottom_mul: float = 0.0      # default: NO bottom radiation
                 ):

        # --- Geometry / material ---
        self.L = float(seg_len)
        self.W = float(seg_width)
        self.H = float(seg_height)
        self.rho, self.cp, self.k = float(rho), float(cp), float(k)

        # --- Environment & time ---
        self.T_bed, self.T_amb, self.h = float(T_bed), float(T_amb), float(h)
        self.dt, self.post_cooldown = float(dt), float(post_cooldown)
        self.T_nozzle = float(T_nozzle)

        # --- Linking radii ---
        self.link_max = float(link_max)  # same-layer end-to-end
        self.vert_rad = float(0.6*self.W) if vert_radius is None else float(vert_radius)  # vertical search

        # --- Contact scalars (areas) ---
        self.bed_contact_frac  = float(bed_contact_frac)
        self.contact_area_frac = float(contact_area_frac)

        # --- Convection per-face multipliers ---
        self.h_top_mul    = float(h_top_mul)
        self.h_side_mul   = float(h_side_mul)
        self.h_bottom_mul = float(h_bottom_mul)

        # --- Radiation params & per-face multipliers (independent of h_*) ---
        self.enable_radiation = bool(enable_radiation)
        self.emissivity = float(emissivity)
        self.rad_top_mul    = float(rad_top_mul)
        self.rad_side_mul   = float(rad_side_mul)
        self.rad_bottom_mul = float(rad_bottom_mul)  # currently unused (we omit bottom radiation)
        self.sigma = 5.670374419e-8  # Stefan–Boltzmann constant (W/m²·K⁴)

        # --- Lump geometry (constant size thanks to uniform resampling) ---
        self.V = self.W * self.H * self.L
        self.C = self.rho * self.cp * self.V
        self.A_end    = self.W * self.H
        self.A_top    = self.W * self.L
        self.A_side   = self.H * self.L
        self.A_bottom = self.W * self.L

        # --- Conductances ---
        # Bed coupling (effective solid link through half thickness), scaled by bottom multiplier.
        self.G_bed_contact = self.h_bottom_mul * (
            (self.k * (self.bed_contact_frac * self.A_bottom)) / (self.H/2.0 + 1e-12)
        )

        # Convection (exposed surfaces) — scaled by per-face multipliers.
        # We treat the two long sides and the two free ends as “side-like” for the multiplier.
        self.G_top      = (self.h * self.h_top_mul)  * self.A_top            # top convection (later gated by coverage)
        self.G_sides    = (self.h * self.h_side_mul) * (2.0 * self.A_side)   # two long sides
        self.G_endfilm  = (self.h * self.h_side_mul) * self.A_end            # a single end face

        # Solid links
        self.G_end_pair  = self.k * self.A_end / max(self.L, 1e-12)  # same-layer end contact
        self.G_vert_pair = (self.k * (self.contact_area_frac * self.A_bottom)) / max(self.H, 1e-12)  # layer-to-layer per unit fraction

        # --- State ---
        self.pos   = []      # [ [x,y,zc], ... ]
        self.T     = []      # temperatures
        self.Genv  = []      # convection to ambient per lump (sum of exposed faces)
        self.Gbed  = []      # bed conductance (layer 0 only)
        self.edges = []      # list of (i, j, G) solid/contact links

        self.layer_bins   = {}   # layer index -> list of lump indices
        self.top_cover    = []   # per-lump TOP coverage fraction (0..1)
        self.bot_alloc    = []   # per-lump BOTTOM allocation used by vertical links (0..1)

        self.times = [0.0]
        self.Tavg  = [None]

        self.microgrid_enable = bool(microgrid_enable)
        self.nx = int(nx); self.nz = int(nz)

    # -------- helpers --------
    def alpha(self):
        return self.k/(self.rho*self.cp)

    def layer_of(self, zc):
        return int(np.floor(zc / self.H + 1e-6))

    def _maybe_link_horizontal(self, i_new):
        """Connect consecutive lumps within a layer (end-to-end)."""
        if i_new == 0:
            return
        i_prev = i_new - 1
        dxy = np.linalg.norm(self.pos[i_new][:2] - self.pos[i_prev][:2])
        if dxy <= self.link_max:
            # shared end becomes solid → remove one end film on each (use side multiplier-scaled end film)
            self.Genv[i_prev] = max(0.0, self.Genv[i_prev] - self.G_endfilm)
            self.Genv[i_new]  = max(0.0, self.Genv[i_new]  - self.G_endfilm)
            self.edges.append((i_prev, i_new, self.G_end_pair))

    def _weight(self, d):
        """Overlap weight w(d) in [0,1]: full at d=0, zero at d>=W/2."""
        r = 0.5 * self.W
        if d >= r:
            return 0.0
        return 1.0 - d/r

    def _maybe_link_vertical(self, i_new):
        """
        Many-to-many, area-weighted vertical contact:
        - find ALL candidates below within vert_rad
        - compute weights w_j = w(d_j)
        - normalize so sum w_j <= 1
        - allocate phi_j = min(w_j * (1 - bot_alloc[i_new]), 1 - top_cover[j])
        - add edges with G = phi_j * G_vert_pair
        - reduce lower lump's TOP convection by phi_j * G_top (already scaled by h_top_mul)
        """
        p = self.pos[i_new]
        lay = self.layer_of(p[2])

        # layer 0 → bed contact
        if lay == 0:
            self.Gbed[i_new] = self.G_bed_contact
            return

        cand = self.layer_bins.get(lay - 1, [])
        if not cand:
            return

        xy = p[:2]
        # collect neighbors within vertical radius
        nbrs = []
        for j in cand[-800:]:  # locality window for speed
            d = np.linalg.norm(xy - self.pos[j][:2])
            if d <= self.vert_rad:
                w = self._weight(d)
                if w > 0.0:
                    nbrs.append((j, w))

        if not nbrs:
            return

        # normalize weights so total <= 1
        wsum = sum(w for _, w in nbrs)
        if wsum > 1e-12:
            nbrs = [(j, w/wsum) for (j, w) in nbrs]

        # remaining fractions
        rem_upper = max(0.0, 1.0 - self.bot_alloc[i_new])

        for j, wj in nbrs:
            if rem_upper <= 1e-12:
                break
            rem_lower = max(0.0, 1.0 - self.top_cover[j])
            if rem_lower <= 1e-12:
                continue

            # proposed fraction for this pair
            phi = min(wj * rem_upper, rem_lower)
            if phi <= 1e-12:
                continue

            # add partial vertical conductance; reduce top convection of lower
            self.edges.append((j, i_new, self.G_vert_pair * phi))
            # IMPORTANT: subtract the *effective* top convection (already scaled by h_top_mul)
            self.Genv[j] = max(0.0, self.Genv[j] - self.G_top * phi)

            # update coverage trackers
            self.top_cover[j]  += phi
            self.bot_alloc[i_new] += phi
            rem_upper -= phi

            if self.bot_alloc[i_new] >= 1.0 - 1e-12:
                break

    # -------- public API --------
    def add_lump(self, xyz, T_init=None):
        if T_init is None:
            T_init = self.T_nozzle
        i = len(self.T)
        self.T.append(float(T_init))
        self.pos.append(np.asarray(xyz, float))

        # initially exposed: TOP + two SIDES + two free ENDS (all already scaled by multipliers)
        self.Genv.append(self.G_top + self.G_sides + 2.0*self.G_endfilm)
        self.Gbed.append(0.0)
        self.top_cover.append(0.0)
        self.bot_alloc.append(0.0)

        # register in layer bin
        lay = self.layer_of(xyz[2])
        self.layer_bins.setdefault(lay, []).append(i)

        # links
        self._maybe_link_horizontal(i)
        self._maybe_link_vertical(i)

    def step(self, dt):
        """Forward Euler update."""
        if len(self.T) == 0:
            self.times.append(self.times[-1] + dt)
            self.Tavg.append(None)
            return

        T = np.asarray(self.T, float)
        net = np.zeros_like(T)

        # solid/contact edges
        for (i, j, G) in self.edges:
            dT = T[j] - T[i]
            net[i] += G * dT
            net[j] -= G * dT

        # boundary exchanges (ambient + bed) — convection currently stored in Genv,
        # which already reflects coverage + per-face multipliers + end removal.
        Genv = np.asarray(self.Genv, float)
        Gbed = np.asarray(self.Gbed, float)
        net += Genv * (self.T_amb - T) + Gbed * (self.T_bed - T)

        # ---- Radiation (optional; independent of convection multipliers) ----
        # Linearized about current temperature; acts like a temperature-dependent conductance.
        if self.enable_radiation and self.emissivity > 0.0:
            # Kelvin temps
            T_K = T + 273.15
            T_ambK = self.T_amb + 273.15

            # Effective radiating areas per lump:
            #  - TOP: gated by exposure (1 - top_cover) and scaled by rad_top_mul
            #  - SIDES: two sides scaled by rad_side_mul
            #  - ENDS: number of free ends inferred from Genv baseline, scaled by rad_side_mul
            #  NOTE: bottom radiation is intentionally omitted (rad_bottom_mul unused).
            top_exposure = np.clip(1.0 - np.asarray(self.top_cover, float), 0.0, 1.0)
            A_top_eff  = self.A_top  * top_exposure * self.rad_top_mul
            A_side_eff = (2.0 * self.A_side) * self.rad_side_mul

            # Estimate number of free ends now (0..2) from Genv baseline:
            base_top_sides = self.G_top + self.G_sides
            ends_raw = np.maximum(0.0, (Genv - base_top_sides)) / max(self.G_endfilm, 1e-12)
            n_ends_eff = np.clip(ends_raw, 0.0, 2.0)  # fractional is fine
            A_ends_eff = n_ends_eff * self.A_end * self.rad_side_mul

            A_eff = A_top_eff + A_side_eff + A_ends_eff

            # G_rad(T) = ε σ 4 T^3 A_eff  (linearization of σ (T^4 - T_amb^4))
            G_rad = self.emissivity * self.sigma * 4.0 * (T_K**3) * A_eff
            # contribution to net (toward ambient in K)
            net += G_rad * (T_ambK - T_K)
        # --------------------------------------------------------------------

        # update temperatures
        T += (dt / self.C) * net
        self.T = T.tolist()
        self.times.append(self.times[-1] + dt)
        self.Tavg.append(float(T.mean()))
