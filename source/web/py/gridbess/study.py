"""
A study = one network + one period of weather + battery settings.
Mirrors scenarios/run_bus_scenarios.py:

  baseline time series (no battery)                  -> once, reused by every placement
  for a battery bus:
     model of the grid's response to the battery     -> tangent sensitivities (peak hour
                                                         or every hour) or piecewise table
     optimise schedule (objective 1 LP / 3 QP)
     validate the schedule with a full AC time series
     metrics over the WHOLE network
"""
import math
import time

import numpy as np

from . import powerflow as pf
from .network import Network
from .optimize import optimise_sensitivity, optimise_piecewise, element_weights
from .templates import DEFAULT_PROFILES
from .weather import ghi_to_pv_fraction, wind_speed_to_wp_fraction

DEFAULTS = {
    "battery_mw": 4.0, "battery_mwh": 16.0, "eta_c": 0.95, "eta_d": 0.95, "soc_init": 0.5, "dt_h": 1.0,
    "probe_mw": 2.0, "n_levels": 9, "weight_line": 1.0, "weight_trafo": 1.0, "element_weights": None,
}
MODELS = ("tangent", "tangent_hourly", "piecewise")
OBJECTIVES = ("minmax", "quadratic_peak")


def physical_soc(schedule, mwh, eta_c, eta_d, soc_init, dt_h):
    """SOC a real battery would follow for a net schedule (+ = discharge)."""
    out = np.empty(len(schedule))
    soc = soc_init * mwh
    for t, p in enumerate(schedule):
        soc += (-p * dt_h / eta_d) if p >= 0 else (-p * dt_h * eta_c)
        out[t] = soc
    return out


class Study:
    def __init__(self, spec, ghi, wind, settings=None, progress=None):
        self.s = dict(DEFAULTS); self.s.update({k: v for k, v in (settings or {}).items() if v is not None})
        self.net = Network(spec)
        self.T = int(min(len(ghi), len(wind)))
        if self.T < 24:
            raise ValueError("The period must be at least one day.")
        self.progress = progress or (lambda stage, frac: None)
        self.pv = ghi_to_pv_fraction(ghi[:self.T])
        self.wp = wind_speed_to_wp_fraction(wind[:self.T])
        profiles = dict(DEFAULT_PROFILES); profiles.update(spec.get("profiles") or {})
        hours = np.arange(self.T) % 24
        cols = []
        for name in self.net.load_profile:
            prof = profiles.get(name)
            if prof is None or len(prof) != 24:
                raise ValueError(f"Load profile '{name}' needs 24 hourly values.")
            cols.append(np.asarray(prof, float)[hours])
        self.load_scale = np.stack(cols, axis=1) if cols else np.zeros((self.T, 0))
        sc = []
        for typ in self.net.sgen_type:
            sc.append(self.pv if typ == "PV" else self.wp if typ in ("WP", "WIND") else np.ones(self.T))
        self.sgen_scale = np.stack(sc, axis=1) if sc else np.zeros((self.T, 0))
        self.bat = {"mw": float(self.s["battery_mw"]), "mwh": float(self.s["battery_mwh"]),
                    "eta_c": float(self.s["eta_c"]), "eta_d": float(self.s["eta_d"]),
                    "soc_init": float(self.s["soc_init"]), "dt_h": float(self.s["dt_h"])}
        self.weights = element_weights(self.net, float(self.s["weight_line"]), float(self.s["weight_trafo"]),
                                       self.s.get("element_weights"))
        self.P0, self.Q0 = pf.injections(self.net, self.load_scale, self.sgen_scale)
        self.base = None

    # --------------------------------------------------------------- time series
    def _run(self, bus_idx=None, p=None, rows=None):
        P, Q = self.P0, self.Q0
        if rows is not None:
            P, Q = P[rows], Q[rows]
        if bus_idx is not None:
            P = P.copy(); P[:, bus_idx] += p
        return pf.solve(self.net, P, Q)

    def baseline(self):
        if self.base is None:
            t0 = time.time()
            self.base = self._run()
            self.base["seconds"] = time.time() - t0
        return self.base

    # ------------------------------------------------------------------ one bus
    def run_bus(self, bus_id, objective="minmax", model="tangent", keep_series=True, table_cache=None):
        assert objective in OBJECTIVES and model in MODELS
        net, s, bat, T = self.net, self.s, self.bat, self.T
        base = self.baseline()
        L0 = base["loading"]                                   # (T, L)
        b = net.bus_pos[int(bus_id)]
        path = net.path_elements(bus_id)
        timing = {}
        t0 = time.time()

        if model == "piecewise":
            levels = np.linspace(-bat["mw"], bat["mw"], int(s["n_levels"]))
            table = np.empty((T, len(levels), len(net.elements)))
            for k, lev in enumerate(levels):
                self.progress("Measuring the grid response", k / len(levels))
                table[:, k, :] = L0 if lev == 0 else self._run(b, lev)["loading"]
            timing["model_s"] = time.time() - t0; t0 = time.time()
            self.progress("Optimising the schedule", 0.0)
            d, c, soc_model, peak_model = optimise_piecewise(objective, table, levels, bat, self.weights)
        else:
            probe = float(s["probe_mw"])
            if model == "tangent":
                score = L0[:, path].sum(axis=1) if path else L0.sum(axis=1)
                peak_hour = int(np.argmax(score))
                rows = [peak_hour] * 3
                P = self.P0[rows].copy(); P[1, b] += probe; P[2, b] -= probe
                r = pf.solve(net, P, self.Q0[rows])["loading"]
                s_d = (r[1] - r[0]) / probe; s_c = (r[2] - r[0]) / probe
            else:
                self.progress("Measuring hourly sensitivities", 0.0)
                up = self._run(b, probe)["loading"]
                self.progress("Measuring hourly sensitivities", 0.5)
                dn = self._run(b, -probe)["loading"]
                s_d = (up - L0) / probe; s_c = (dn - L0) / probe
            timing["model_s"] = time.time() - t0; t0 = time.time()
            self.progress("Optimising the schedule", 0.0)
            d, c, soc_model, peak_model = optimise_sensitivity(objective, L0, s_d, s_c, bat, self.weights)
        timing["optimise_s"] = time.time() - t0; t0 = time.time()

        schedule = np.where(np.abs(d - c) < 1e-7, 0.0, d - c)
        self.progress("Validating with AC load flow", 0.0)
        val = self._run(b, schedule)
        timing["validate_s"] = time.time() - t0
        return self._metrics(bus_id, objective, model, path, schedule, d, c, peak_model, val, timing, keep_series)

    def _metrics(self, bus_id, objective, model, path, schedule, d, c, peak_model, val, timing, keep_series):
        net, base, bat = self.net, self.base, self.bat
        L0, L1 = base["loading"], val["loading"]
        pk0, pk1 = L0.max(axis=0), L1.max(axis=0)
        w = self.weights
        tso0, tso1 = base["slack_p_mw"], val["slack_p_mw"]
        soc = physical_soc(schedule, bat["mwh"], bat["eta_c"], bat["eta_d"], bat["soc_init"], bat["dt_h"])
        res = {
            "bus": int(bus_id), "bus_name": net.bus_names[net.bus_pos[int(bus_id)]],
            "objective": objective, "model": model,
            "path": [int(k) for k in path],
            "max_loading_before_pct": float(pk0.max()), "max_loading_after_pct": float(pk1.max()),
            "mean_of_max_before_pct": float(pk0.mean()), "mean_of_max_after_pct": float(pk1.mean()),
            "mean_loading_before_pct": float(L0.mean()), "mean_loading_after_pct": float(L1.mean()),
            "mean_of_max_path_before_pct": float(pk0[path].mean()) if path else float("nan"),
            "mean_of_max_path_after_pct": float(pk1[path].mean()) if path else float("nan"),
            "weighted_overload_before": float((w * pk0 ** 2).sum()),
            "weighted_overload_after": float((w * pk1 ** 2).sum()),
            "worst_element_increase_pct": float((pk1 - pk0).max()),
            "hours_overloaded_before": int((L0 > 100).any(axis=1).sum()),
            "hours_overloaded_after": int((L1 > 100).any(axis=1).sum()),
            "vm_min_before": float(base["vm_pu"].min()), "vm_min_after": float(val["vm_pu"].min()),
            "vm_max_before": float(base["vm_pu"].max()), "vm_max_after": float(val["vm_pu"].max()),
            "tso_std_before_mw": float(tso0.std(ddof=1)), "tso_std_after_mw": float(tso1.std(ddof=1)),
            "tso_ptp_before_mw": float(tso0.max() - tso0.min()), "tso_ptp_after_mw": float(tso1.max() - tso1.min()),
            "energy_discharged_mwh": float(np.clip(schedule, 0, None).sum() * bat["dt_h"]),
            "energy_charged_mwh": float(np.clip(-schedule, 0, None).sum() * bat["dt_h"]),
            "soc_min_mwh": float(soc.min()), "soc_max_mwh": float(soc.max()), "soc_end_mwh": float(soc[-1]),
            "simultaneous_mwh": float(np.minimum(d, c).sum() * bat["dt_h"]),
            "peaks_before": pk0.round(4).tolist(), "peaks_after": pk1.round(4).tolist(),
            "peaks_model": np.asarray(peak_model).round(4).tolist(),
            "timing": timing,
        }
        res["equivalent_cycles"] = res["energy_discharged_mwh"] / bat["mwh"] if bat["mwh"] else 0.0
        if keep_series:
            res["schedule_mw"] = schedule.round(5).tolist()
            res["soc_mwh"] = soc.round(5).tolist()
            res["loading_after"] = L1.round(3).tolist()
            res["vm_after"] = val["vm_pu"].round(5).tolist()
            res["tso_after_mw"] = tso1.round(4).tolist()
        return res

    # ------------------------------------------------------------ all placements
    def run_all(self, objective="minmax", model="tangent", buses=None):
        buses = buses or self.net.battery_candidates()
        self.baseline()
        out = []
        outer = self.progress
        for i, bus in enumerate(buses):
            def inner(stage, frac, i=i, bus=bus):
                outer(f"Bus {self.net.bus_names[self.net.bus_pos[int(bus)]]}: {stage}", (i + frac) / len(buses))
            self.progress = inner
            try:
                out.append(self.run_bus(bus, objective, model, keep_series=False))
            except Exception as ex:                         # keep going, report per bus
                out.append({"bus": int(bus), "bus_name": self.net.bus_names[self.net.bus_pos[int(bus)]],
                            "error": str(ex)})
            finally:
                self.progress = outer
        return out

    def baseline_summary(self):
        base = self.baseline()
        return {
            "hours": self.T,
            "elements": [{"type": e[0], "id": e[1], "name": nme} for e, nme in zip(self.net.elements, self.net.element_names)],
            "buses": [{"id": bid, "name": nm} for bid, nm in zip(self.net.bus_ids, self.net.bus_names)],
            "candidates": self.net.battery_candidates(),
            "loading": base["loading"].round(3).tolist(),
            "vm": base["vm_pu"].round(5).tolist(),
            "tso_mw": base["slack_p_mw"].round(4).tolist(),
            "load_mw": (self.load_scale * self.net.load_p).sum(axis=1).round(4).tolist() if len(self.net.loads) else [0.0] * self.T,
            "pv_mw": (self.sgen_scale * self.net.sgen_p)[:, [i for i, t in enumerate(self.net.sgen_type) if t == "PV"]].sum(axis=1).round(4).tolist(),
            "wind_mw": (self.sgen_scale * self.net.sgen_p)[:, [i for i, t in enumerate(self.net.sgen_type) if t in ("WP", "WIND")]].sum(axis=1).round(4).tolist(),
            "seconds": base["seconds"],
        }
