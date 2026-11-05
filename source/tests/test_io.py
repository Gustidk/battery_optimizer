import os
import sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "../web/py")
import pandapower as pp
from gridbess.io_pandapower import import_pandapower_json, to_pandapower_script
from gridbess.network import Network
from gridbess import powerflow as pf
for fn in ["tests/cigre_user.json", "tests/cigre_pn.json", "tests/pandapower_balanced.json"]:
    try:
        txt = open(fn).read()
    except FileNotFoundError:
        print("missing", fn); continue
    spec, w = import_pandapower_json(txt)
    net = pp.from_json(fn)
    has_unsupported = any(len(net[k][net[k].in_service]) for k in ("gen", "storage") if k in net)
    pp.runpp(net, calculate_voltage_angles=True)
    g = Network(spec)
    P, Q = pf.injections(g, np.ones((1, len(g.loads))), np.ones((1, len(g.sgens))))
    r = pf.solve(g, P, Q)
    ll = {l["id"]: v for l, v in zip(spec["lines"], r["line_loading"][0])}
    dl = max(abs(net.res_line.loading_percent[i] - v) for i, v in ll.items())
    print(f"{fn}: {len(spec['buses'])} buses, {len(spec['lines'])} lines, {len(spec['trafos'])} trafos; warnings={w}")
    print(f"   max line loading deviation vs pandapower: {dl:.2e}  (unsupported elements present: {has_unsupported})")
    # round trip via generated script
    ns = {}
    exec(to_pandapower_script(spec), ns)
    n2 = ns["net"]
    print(f"   script round-trip: max line loading {n2.res_line.loading_percent.max():.4f} vs ours {r['line_loading'].max():.4f}")
