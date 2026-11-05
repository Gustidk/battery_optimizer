import os
"""Validate gridbess network + batched NR against the user's lightsim2grid code and pandapower."""
import sys, copy, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "../web/py"); sys.path.insert(0, os.environ.get("ORIGINAL_CODE", "../original"))
import pandapower as pp
from gridbess.network import Network
from gridbess import powerflow as pf
from gridbess.templates import cigre_mv

# --- 1) CIGRE vs the user's build_network + FastPowerFlow over 500 random hours
from network.build_cigre_mv import build_network
from network.fast_powerflow import FastPowerFlow
net0, bus_map, lpt = build_network()
fpf = FastPowerFlow(net0)
g = Network(cigre_mv())
rng = np.random.default_rng(3); T = 500
ls = rng.uniform(0.1, 1.3, (T, len(g.loads))); ss = rng.uniform(0, 1, (T, len(g.sgens)))
P, Q = pf.injections(g, ls, ss)
r = pf.solve(g, P, Q)
ref_l, ref_t, ref_v, ref_ph = [], [], [], []
bp, bq, bs = net0.load.p_mw.values.copy(), net0.load.q_mvar.values.copy(), net0.sgen.p_mw.values.copy()
for t in range(T):
    net0.load.p_mw.values[:] = bp * ls[t]; net0.load.q_mvar.values[:] = bq * ls[t]; net0.sgen.p_mw.values[:] = bs * ss[t]
    fpf.run(net0)
    ref_l.append(fpf.line_loading.copy()); ref_t.append(fpf.trafo_loading.copy()); ref_v.append(fpf.vm_pu.copy()); ref_ph.append(np.array(fpf.trafo_p_hv))
print("CIGRE vs FastPowerFlow: line %.1e  trafo %.1e  vm %.1e  p_hv %.1e" % (
    np.abs(np.array(ref_l) - r["line_loading"]).max(), np.abs(np.array(ref_t) - r["trafo_loading"]).max(),
    np.abs(np.array(ref_v) - r["vm_pu"]).max(), np.abs(np.array(ref_ph) - r["trafo_p_hv_mw"]).max()))

# --- 2) random radial/meshed nets with all supported features vs pandapower.runpp
def random_spec(seed):
    rng = np.random.default_rng(seed)
    nb = rng.integers(5, 30)
    buses = [{"id": 100, "name": "HV", "vn_kv": 110.0}, {"id": 101, "name": "HV2", "vn_kv": 110.0}]
    buses += [{"id": i, "name": f"B{i}", "vn_kv": 20.0} for i in range(nb)]
    lines = []
    for i in range(1, nb):
        j = int(rng.integers(0, i))
        lines.append({"id": len(lines), "name": f"L{i}", "from_bus": j, "to_bus": i, "length_km": float(rng.uniform(.2, 5)),
                      "r_ohm_per_km": float(rng.uniform(.1, .6)), "x_ohm_per_km": float(rng.uniform(.1, .5)),
                      "c_nf_per_km": float(rng.uniform(0, 300)), "g_us_per_km": float(rng.choice([0, 0.5])),
                      "max_i_ka": float(rng.uniform(.2, .4)), "parallel": int(rng.choice([1, 1, 2])), "df": float(rng.choice([1, .9]))})
    sw = []
    for k in range(int(rng.integers(0, 4))):                      # ring closures, some open
        a, b = rng.choice(nb, 2, replace=False)
        lines.append({"id": len(lines), "name": f"R{k}", "from_bus": int(a), "to_bus": int(b), "length_km": 1.0,
                      "r_ohm_per_km": .3, "x_ohm_per_km": .3, "c_nf_per_km": 200.0, "max_i_ka": .3})
        if rng.random() < .7:
            sw.append({"bus": int(a if rng.random() < .5 else b), "line": len(lines) - 1, "closed": False})
    trafos = [{"id": 0, "name": "T1", "hv_bus": 100, "lv_bus": 0, "sn_mva": float(rng.choice([25, 40])), "vn_hv_kv": 110.0,
               "vn_lv_kv": float(rng.choice([20, 21])), "vk_percent": float(rng.uniform(8, 14)), "vkr_percent": float(rng.uniform(.2, .6)),
               "pfe_kw": float(rng.choice([0, 20])), "i0_percent": float(rng.choice([0, .07])), "shift_degree": float(rng.choice([0, 30, 150])),
               "tap_side": str(rng.choice(["hv", "lv"])), "tap_pos": int(rng.integers(-3, 4)), "tap_neutral": 0,
               "tap_step_percent": 1.5, "parallel": int(rng.choice([1, 2]))},
              {"id": 1, "name": "T2", "hv_bus": 100, "lv_bus": 101, "sn_mva": 100.0, "vn_hv_kv": 110.0, "vn_lv_kv": 110.0,
               "vk_percent": 10.0, "vkr_percent": .3, "pfe_kw": 0, "i0_percent": 0, "shift_degree": 0}]
    loads = [{"id": i, "bus": int(b), "p_mw": float(rng.uniform(.1, 1.5)), "q_mvar": float(rng.uniform(0, .4))}
             for i, b in enumerate(rng.choice(nb, nb))]
    sgens = [{"id": i, "bus": int(b), "p_mw": float(rng.uniform(0, 2)), "q_mvar": float(rng.uniform(-.2, .2)), "type": "PV"}
             for i, b in enumerate(rng.choice(nb, nb // 3))]
    shunts = [{"bus": int(b), "p_mw": float(rng.uniform(0, .05)), "q_mvar": float(rng.uniform(-1, 1)), "step": int(rng.integers(1, 3))}
              for b in rng.choice(nb, int(rng.integers(0, 3)))]
    ext = [{"bus": 100, "vm_pu": float(rng.uniform(.98, 1.04)), "va_degree": float(rng.choice([0, 10]))}]
    return {"buses": buses, "lines": lines, "trafos": trafos, "loads": loads, "sgens": sgens, "switches": sw,
            "ext_grids": ext, "shunts": shunts, "f_hz": 50, "sn_mva": float(rng.choice([1, 10]))}

def to_pp(spec):
    net = pp.create_empty_network(f_hz=spec["f_hz"], sn_mva=spec["sn_mva"])
    for b in spec["buses"]: pp.create_bus(net, vn_kv=b["vn_kv"], index=b["id"])
    for e in spec["ext_grids"]: pp.create_ext_grid(net, e["bus"], vm_pu=e["vm_pu"], va_degree=e["va_degree"])
    for l in spec["lines"]:
        pp.create_line_from_parameters(net, l["from_bus"], l["to_bus"], l["length_km"], l["r_ohm_per_km"], l["x_ohm_per_km"],
                                       l["c_nf_per_km"], l["max_i_ka"], g_us_per_km=l.get("g_us_per_km", 0),
                                       parallel=l.get("parallel", 1), df=l.get("df", 1), index=l["id"])
    for s in spec["switches"]: pp.create_switch(net, s["bus"], s["line"], et="l", closed=s["closed"])
    for t in spec["trafos"]:
        kw = {k: t[k] for k in ("tap_side", "tap_pos", "tap_neutral", "tap_step_percent") if k in t}
        if kw: kw["tap_changer_type"] = "Ratio"
        pp.create_transformer_from_parameters(net, t["hv_bus"], t["lv_bus"], t["sn_mva"], t["vn_hv_kv"], t["vn_lv_kv"],
            t["vkr_percent"], t["vk_percent"], t["pfe_kw"], t["i0_percent"], shift_degree=t["shift_degree"],
            parallel=t.get("parallel", 1), index=t["id"], **kw)
    for l in spec["loads"]: pp.create_load(net, l["bus"], l["p_mw"], l["q_mvar"])
    for s in spec["sgens"]: pp.create_sgen(net, s["bus"], s["p_mw"], s["q_mvar"])
    for s in spec["shunts"]: pp.create_shunt(net, s["bus"], q_mvar=s["q_mvar"], p_mw=s["p_mw"], step=s["step"])
    return net

worst = 0
for seed in range(40):
    spec = random_spec(seed)
    net = to_pp(spec)
    try:
        pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10, max_iteration=50)
    except Exception as ex:
        print("seed", seed, "pandapower failed", type(ex).__name__); continue
    g = Network(spec)
    P, Q = pf.injections(g, np.ones((1, len(g.loads))), np.ones((1, len(g.sgens))))
    r = pf.solve(g, P, Q)
    d = max(np.abs(net.res_line.loading_percent.values - r["line_loading"][0]).max(),
            np.abs(net.res_trafo.loading_percent.values - r["trafo_loading"][0]).max(),
            np.abs(net.res_bus.vm_pu.values - r["vm_pu"][0]).max() * 100,
            abs(net.res_ext_grid.p_mw.sum() - r["slack_p_mw"][0]))
    worst = max(worst, d)
    if d > 1e-5: print("seed", seed, "MISMATCH", d)
print("random nets vs pandapower: worst abs deviation %.2e" % worst)
