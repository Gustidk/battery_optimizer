"""
Network model: turns a JSON network description ("spec") into the matrices a
load flow needs (Ybus, Yf, Yt) -- built with exactly the same per-unit
formulas as pandapower (lines: pi-model, transformers: t-model), so results
match pandapower / lightsim2grid.

Supported: buses, ext_grids (slack), lines, 2-winding transformers (with fixed
tap position and phase shift), line switches (open = line ends blind at that
bus, like pandapower), constant-power loads and static generators.

Spec layout (all lists of dicts, ids are arbitrary integers):
    buses     {id, name, vn_kv, x?, y?}
    ext_grids {bus, vm_pu=1.0, va_degree=0}
    lines     {id, name, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km,
               c_nf_per_km=0, g_us_per_km=0, max_i_ka, parallel=1, df=1}
    trafos    {id, name, hv_bus, lv_bus, sn_mva, vn_hv_kv, vn_lv_kv, vk_percent,
               vkr_percent, pfe_kw=0, i0_percent=0, shift_degree=0,
               tap_side=None, tap_pos=0, tap_neutral=0, tap_step_percent=0,
               parallel=1, df=1}
    switches  {bus, line, closed}
    shunts    {bus, p_mw=0, q_mvar, step=1}
    loads     {id, name, bus, p_mw, q_mvar, profile}
    sgens     {id, name, bus, p_mw, q_mvar=0, type}   type: "PV" | "WP"
    f_hz=50, sn_mva=1
"""
import math
from collections import deque

import numpy as np


class NetworkError(ValueError):
    """Raised with a message meant for the person who built the network."""


def _g(d, key, default=None):
    v = d.get(key, default)
    return default if v is None else v


class Network:
    def __init__(self, spec):
        self.spec = spec
        self.f_hz = float(_g(spec, "f_hz", 50.0))
        self.sn_mva = float(_g(spec, "sn_mva", 1.0))
        buses = spec.get("buses") or []
        if not buses:
            raise NetworkError("The network has no buses.")
        self.bus_ids = [int(b["id"]) for b in buses]
        if len(set(self.bus_ids)) != len(self.bus_ids):
            raise NetworkError("Two buses share the same id.")
        self.bus_pos = {b: i for i, b in enumerate(self.bus_ids)}
        self.bus_names = [str(_g(b, "name", f"Bus {b['id']}")) for b in buses]
        self.n_bus = len(buses)                      # real buses (aux buses come after)
        vn = [float(b["vn_kv"]) for b in buses]

        self.lines = list(spec.get("lines") or [])
        self.trafos = list(spec.get("trafos") or [])
        self.loads = list(spec.get("loads") or [])
        self.sgens = list(spec.get("sgens") or [])
        ext = list(spec.get("ext_grids") or [])
        if not ext:
            raise NetworkError("Add an external grid (slack bus) so the network has a voltage reference.")

        def pos(bus_id, what):
            try:
                return self.bus_pos[int(bus_id)]
            except (KeyError, TypeError, ValueError):
                raise NetworkError(f"{what} refers to bus {bus_id}, which does not exist.")

        # --- open line switches: move that line end to a new, empty aux bus ---
        line_by_id = {int(l["id"]): k for k, l in enumerate(self.lines)}
        line_from = [pos(l["from_bus"], f"Line '{l.get('name', l['id'])}'") for l in self.lines]
        line_to = [pos(l["to_bus"], f"Line '{l.get('name', l['id'])}'") for l in self.lines]
        self.line_open_end = [None] * len(self.lines)
        for sw in spec.get("switches") or []:
            if bool(_g(sw, "closed", True)):
                continue
            k = line_by_id.get(int(sw["line"]))
            if k is None:
                raise NetworkError(f"A switch refers to line {sw['line']}, which does not exist.")
            b = pos(sw["bus"], "A switch")
            if b == line_from[k]:
                end = "from"
            elif b == line_to[k]:
                end = "to"
            else:
                raise NetworkError(f"Switch at bus {sw['bus']} is not at an end of line {sw['line']}.")
            aux = len(vn)
            vn.append(vn[b])
            if end == "from":
                line_from[k] = aux
            else:
                line_to[k] = aux
            self.line_open_end[k] = end
        self.vn_kv = np.array(vn)
        n = len(vn)
        self.n = n

        # --- branch admittances (pandapower / pypower conventions) ---
        nbr = len(self.lines) + len(self.trafos)
        f = np.zeros(nbr, dtype=int); t = np.zeros(nbr, dtype=int)
        Yff = np.zeros(nbr, complex); Yft = np.zeros(nbr, complex)
        Ytf = np.zeros(nbr, complex); Ytt = np.zeros(nbr, complex)
        sn = self.sn_mva
        for k, l in enumerate(self.lines):
            fb, tb = line_from[k], line_to[k]
            if fb == tb:
                raise NetworkError(f"Line '{l.get('name', l['id'])}' starts and ends at the same bus.")
            length = float(l["length_km"]); par = float(_g(l, "parallel", 1))
            base_z = vn[fb] ** 2 / sn
            r = float(l["r_ohm_per_km"]) * length / base_z / par
            x = float(l["x_ohm_per_km"]) * length / base_z / par
            if r == 0 and x == 0:
                raise NetworkError(f"Line '{l.get('name', l['id'])}' has zero impedance.")
            b = 2 * self.f_hz * math.pi * float(_g(l, "c_nf_per_km", 0)) * 1e-9 * base_z * length * par
            g = float(_g(l, "g_us_per_km", 0)) * 1e-6 * base_z * length * par
            ys = 1 / (r + 1j * x); bc = g + 1j * b
            f[k], t[k] = fb, tb
            Yff[k] = ys + bc / 2; Ytt[k] = ys + bc / 2; Yft[k] = -ys; Ytf[k] = -ys
        nl = len(self.lines)
        for j, tr in enumerate(self.trafos):
            k = nl + j
            name = tr.get("name", tr["id"])
            hb = pos(tr["hv_bus"], f"Transformer '{name}'"); lb = pos(tr["lv_bus"], f"Transformer '{name}'")
            vnh = float(tr["vn_hv_kv"]); vnl = float(tr["vn_lv_kv"])
            vnl_raw = vnl
            side = _g(tr, "tap_side", None)
            step = float(_g(tr, "tap_step_percent", 0) or 0)
            steps = float(_g(tr, "tap_pos", 0) or 0) - float(_g(tr, "tap_neutral", 0) or 0)
            if side in ("hv", "lv") and step and steps:
                fac = 1 + steps * step / 100
                if side == "hv":
                    vnh *= fac
                else:
                    vnl *= fac
            sn_t = float(tr["sn_mva"]); par = float(_g(tr, "parallel", 1))
            vk = float(tr["vk_percent"]); vkr = float(tr["vkr_percent"])
            if vk <= 0 or vkr < 0 or vkr > vk:
                raise NetworkError(f"Transformer '{name}': need 0 <= vkr_percent <= vk_percent and vk_percent > 0.")
            vn_lv_bus = vn[lb]; vn_hv_bus = vn[hb]
            tap_lv = (vnl / vn_lv_bus) ** 2 * sn
            z = vk / 100 / sn_t * tap_lv; r = vkr / 100 / sn_t * tap_lv
            x = math.sqrt(max(z * z - r * r, 0.0))
            r /= par; x /= par
            base_z = vn_lv_bus ** 2 / sn
            pfe = float(_g(tr, "pfe_kw", 0)) * 1e-3
            ym = float(_g(tr, "i0_percent", 0)) / 100 * sn_t
            b_mva = -math.sqrt(max(ym * ym - pfe * pfe, 0.0))
            scale = base_z * par / (vnl / vnl_raw) ** 2 / vnl_raw ** 2
            g, b = pfe * scale, b_mva * scale
            g_asym = b_asym = 0.0
            if g != 0 or b != 0:                     # t-model -> pi (pandapower _wye_delta)
                za = r * 0.5 + 1j * x * 0.5
                zb = r * 0.5 + 1j * x * 0.5
                zc = 1 / (g + 1j * b)
                zsum = za * zb + za * zc + zb * zc
                zab = zsum / zc; zac = zsum / zb; zbc = zsum / za
                r, x = zab.real, zab.imag
                yf = 1 / zac; yt = 1 / zbc
                g, b = yf.real * 2, yf.imag * 2
                g_asym, b_asym = 2 * yt.real - g, 2 * yt.imag - b
            ratio = (vnh / vnl) / (vn_hv_bus / vn_lv_bus)
            tap = ratio * np.exp(1j * math.pi / 180 * float(_g(tr, "shift_degree", 0)))
            ys = 1 / (r + 1j * x)
            bcf = g + 1j * b; bct = (g + g_asym) + 1j * (b + b_asym)
            f[k], t[k] = hb, lb
            Ytt[k] = ys + bct / 2
            Yff[k] = (ys + bcf / 2) / (tap * np.conj(tap))
            Yft[k] = -ys / np.conj(tap)
            Ytf[k] = -ys / tap
        self.f, self.t = f, t
        Y = np.zeros((n, n), complex)
        np.add.at(Y, (f, f), Yff); np.add.at(Y, (f, t), Yft)
        np.add.at(Y, (t, f), Ytf); np.add.at(Y, (t, t), Ytt)
        # shunts (pandapower: p_mw / q_mvar absorbed at 1 pu, times step)
        self.shunts = list(spec.get("shunts") or [])
        for sh in self.shunts:
            bsh = pos(sh["bus"], "A shunt")
            Y[bsh, bsh] += (float(_g(sh, "p_mw", 0)) - 1j * float(_g(sh, "q_mvar", 0))) * float(_g(sh, "step", 1)) / sn
        self.Y = Y
        r = np.arange(nbr)
        self.Yf = np.zeros((nbr, n), complex); self.Yt = np.zeros((nbr, n), complex)
        self.Yf[r, f] = Yff; self.Yf[r, t] = Yft
        self.Yt[r, f] = Ytf; self.Yt[r, t] = Ytt
        self.line_br = np.arange(nl)
        self.trafo_br = np.arange(nl, nbr)

        # current scaling pu -> kA, and ratings (same as FastPowerFlow / pandapower)
        self.i_scale_f = sn / (math.sqrt(3) * self.vn_kv[f]) if nbr else np.zeros(0)
        self.i_scale_t = sn / (math.sqrt(3) * self.vn_kv[t]) if nbr else np.zeros(0)
        self.line_imax = np.array([float(l["max_i_ka"]) * float(_g(l, "df", 1)) * float(_g(l, "parallel", 1))
                                   for l in self.lines])
        if (self.line_imax <= 0).any():
            raise NetworkError("Every line needs a positive max_i_ka (thermal rating).")
        fac = np.array([float(_g(tr, "parallel", 1)) * float(_g(tr, "df", 1)) for tr in self.trafos])
        sn_t = np.array([float(tr["sn_mva"]) for tr in self.trafos])
        self.trafo_in_hv = sn_t / (math.sqrt(3) * np.array([float(tr["vn_hv_kv"]) for tr in self.trafos])) * fac if len(self.trafos) else np.zeros(0)
        self.trafo_in_lv = sn_t / (math.sqrt(3) * np.array([float(tr["vn_lv_kv"]) for tr in self.trafos])) * fac if len(self.trafos) else np.zeros(0)

        # slack buses
        self.ref = np.array([pos(e["bus"], "The external grid") for e in ext])
        if len(set(self.ref.tolist())) != len(self.ref):
            raise NetworkError("Two external grids sit on the same bus.")
        self.V_ref = np.array([float(_g(e, "vm_pu", 1.0)) * np.exp(1j * math.pi / 180 * float(_g(e, "va_degree", 0)))
                               for e in ext])
        self.pq = np.setdiff1d(np.arange(n), self.ref)
        self._shift = np.zeros(nbr)
        for j, tr in enumerate(self.trafos):
            self._shift[nl + j] = float(_g(tr, "shift_degree", 0))

        # injections: which bus each load / sgen sits on
        self.load_bus = np.array([pos(l["bus"], f"Load '{l.get('name', l.get('id'))}'") for l in self.loads], dtype=int)
        self.sgen_bus = np.array([pos(s["bus"], f"Generator '{s.get('name', s.get('id'))}'") for s in self.sgens], dtype=int)
        self.load_p = np.array([float(l["p_mw"]) for l in self.loads])
        self.load_q = np.array([float(_g(l, "q_mvar", 0)) for l in self.loads])
        self.load_profile = [str(_g(l, "profile", "residential")) for l in self.loads]
        self.sgen_p = np.array([float(s["p_mw"]) for s in self.sgens])
        self.sgen_q = np.array([float(_g(s, "q_mvar", 0)) for s in self.sgens])
        self.sgen_type = [str(_g(s, "type", "PV")).upper() for s in self.sgens]

        # elements the optimisation looks at: every line and every transformer
        self.elements = [("line", int(l["id"])) for l in self.lines] + [("trafo", int(tr["id"])) for tr in self.trafos]
        self.element_names = [str(_g(l, "name", f"Line {l['id']}")) for l in self.lines] + \
                             [str(_g(tr, "name", f"Trafo {tr['id']}")) for tr in self.trafos]
        self._check_connected()
        self.V_init = self._initial_voltage()

    def _initial_voltage(self):
        """Flat start, but with the phase shift of each transformer applied along the
        way from the slack (as pandapower does), so Newton-Raphson starts close."""
        ang = np.full(self.n, np.nan)
        ang[self.ref] = np.angle(self.V_ref)
        adj = self._adjacency()
        q = deque(self.ref.tolist())
        while q:
            u = q.popleft()
            for v, k in adj[u]:
                if np.isnan(ang[v]):
                    sh = math.radians(self._shift[k])
                    ang[v] = ang[u] - sh if u == self.f[k] else ang[u] + sh
                    q.append(v)
        ang = np.nan_to_num(ang)
        V = np.exp(1j * ang)
        V[self.ref] = self.V_ref
        return V

    # ------------------------------------------------------------------ topology
    def _adjacency(self):
        """Graph of closed branches between real buses: bus -> [(neighbour, element_index)]."""
        adj = {i: [] for i in range(self.n)}
        for k in range(len(self.f)):
            a, b = int(self.f[k]), int(self.t[k])
            adj[a].append((b, k)); adj[b].append((a, k))
        return adj

    def _check_connected(self):
        adj = self._adjacency()
        seen = set(self.ref.tolist()); q = deque(seen)
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        missing = [self.bus_names[i] for i in range(self.n_bus) if i not in seen]
        if missing:
            raise NetworkError("These buses are not connected to the external grid: " + ", ".join(missing[:8])
                               + (" ..." if len(missing) > 8 else ""))

    def path_elements(self, bus_id):
        """Branches on the shortest path from the nearest slack to the bus (indices into elements)."""
        target = self.bus_pos[int(bus_id)]
        adj = self._adjacency()
        prev = {int(r): None for r in self.ref}
        q = deque(prev)
        while q:
            u = q.popleft()
            if u == target:
                break
            for v, k in adj[u]:
                if v not in prev:
                    prev[v] = (u, k); q.append(v)
        out = []
        u = target
        while prev.get(u) is not None:
            u, k = prev[u]
            out.append(k)
        return out[::-1]

    def battery_candidates(self):
        """Bus ids where a battery can be placed: every real bus except slack buses."""
        refs = set(self.ref.tolist())
        return [self.bus_ids[i] for i in range(self.n_bus) if i not in refs]
