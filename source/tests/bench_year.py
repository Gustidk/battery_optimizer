import sys,time; sys.path.insert(0,'/home/claude/gridapp/web/py')
from gridbess.study import Study; from gridbess.templates import cigre_mv; from gridbess.weather import synthetic_weather
g,w=synthetic_weather(56.15,10.21,1,8760); st=Study(cigre_mv(),g,w)
t=time.time(); st.baseline(); print('baseline %.1fs'%(time.time()-t), flush=True)
for m in ('tangent','tangent_hourly','piecewise'):
  for o in ('minmax','quadratic_peak'):
    t=time.time(); r=st.run_bus(8,o,m,keep_series=False); print(m,o,'%.1fs'%(time.time()-t), {k:round(v,2) for k,v in r['timing'].items()}, 'mean_of_max %.3f -> %.3f'%(r['mean_of_max_before_pct'],r['mean_of_max_after_pct']), flush=True)
t=time.time(); a=st.run_all('minmax','tangent'); print('all 14 tangent minmax %.1fs'%(time.time()-t), flush=True)
