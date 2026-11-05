import sys,time; sys.path.insert(0,'/home/claude/gridapp/web/py')
from gridbess import optimize as O
from gridbess.study import Study; from gridbess.templates import cigre_mv; from gridbess.weather import synthetic_weather
g,w=synthetic_weather(56.15,10.21,1,8760); st=Study(cigre_mv(),g,w); st.baseline()
for name,opt in [("ipm-nocross",{"solver":"ipm","run_crossover":"off"}),("ipm",{"solver":"ipm"})]:
    O.LP_OPTIONS.clear(); O.LP_OPTIONS.update(opt)
    for m in (sys.argv[1:] or ["piecewise","tangent_hourly"]):
        t=time.time(); r=st.run_bus(8,"minmax",m,keep_series=False)
        print(name, m, "opt %.1fs"%r['timing']['optimise_s'], "AC %.4f"%r['mean_of_max_after_pct'], "simult %.3f MWh"%r['simultaneous_mwh'], flush=True)
