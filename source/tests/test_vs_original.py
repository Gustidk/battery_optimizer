import os
import sys, time, warnings, io, contextlib, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "../web/py"); sys.path.insert(0, os.environ.get("ORIGINAL_CODE", "../original"))
from gridbess.study import Study
from gridbess.templates import cigre_mv
from gridbess.weather import synthetic_weather, ghi_to_pv_fraction, wind_speed_to_wp_fraction
from network.build_cigre_mv import build_network
from profiles.load_profiles import RESIDENTIAL_PROFILE as R, COMMERCIAL_PROFILE as C
from simulation.run_timeseries import run_baseline_timeseries
from scenarios.run_bus_scenarios import run_single_bus_scenario

T = int(sys.argv[1]) if len(sys.argv) > 1 else 168
ghi, wind = synthetic_weather(56.15, 10.21, 152, T)
pv, wp = ghi_to_pv_fraction(ghi), wind_speed_to_wp_fraction(wind)
q = io.StringIO()
with contextlib.redirect_stdout(q):
    net0, bus_map, lpt = build_network()
    out0 = run_baseline_timeseries(net0, lpt, R, C, pv, wp, T)
line0, trafo0, php0 = out0[2], out0[4], out0[7]
st = Study(cigre_mv(), ghi, wind)
st.baseline()
print(f"T={T}")
keys = ["mean_of_max_after_pct", "max_loading_after_pct", "tso_std_after_mw"]
for model, kw in [("tangent", dict(time_varying_sensitivities=False)),
                  ("tangent_hourly", dict(time_varying_sensitivities=True)),
                  ("piecewise", dict(piecewise=True))]:
    for obj in ("minmax", "quadratic_peak"):
        for node in (3, 8, 12):
            t0 = time.time()
            with contextlib.redirect_stdout(q):
                o = run_single_bus_scenario(node, bus_map, R, C, pv, wp, T, line0, trafo0, php0, objective=obj, **kw)
            to = time.time() - t0; t0 = time.time()
            n = st.run_bus(node, obj, model)
            tn = time.time() - t0
            a = [o[k] for k in keys]; b = [n[k] for k in keys]
            print(f"{model:15s} {obj:14s} node {node:2d}: original {a[0]:7.3f} {a[1]:7.3f} {a[2]:6.3f} ({to:5.1f}s) | new {b[0]:7.3f} {b[1]:7.3f} {b[2]:6.3f} ({tn:5.1f}s)  diff {max(abs(x-y) for x,y in zip(a,b)):.3f}")
