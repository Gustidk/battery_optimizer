# Grid-Friendly Battery Planner

Build a distribution network, place a battery, and find the charging schedule that
relieves lines and transformers the most. Everything runs in the visitor's browser:
the Python engine (numpy, scipy, HiGHS, Clarabel) runs in Pyodide (WebAssembly).

## Put it online with GitHub Pages

1. Create a new repository on GitHub, for example `battery-planner`.
2. Upload the **contents** of the `site/` folder to the repository root
   (index.html, worker.js, gridbess.zip, .nojekyll and the vendor folder).
3. In the repository: Settings → Pages → Source: "Deploy from a branch",
   branch `main`, folder `/ (root)`. Save.
4. After a minute the site is at `https://<your-user>.github.io/battery-planner/`.
   A custom domain (e.g. `battery.yourname.dk`) can be set on the same page.

On the hosted site, "Fetch from Open-Meteo" downloads real ERA5 weather directly.

## Try it locally

```
cd site
python -m http.server 8000
```
Open http://localhost:8000 (opening index.html directly as a file does not work,
because the browser blocks workers from `file://`).

## Source layout

```
source/
  web/py/gridbess/      the engine (Python)
    network.py          spec -> Ybus/Yf/Yt, same formulas as pandapower
    powerflow.py        Newton-Raphson vectorised over all hours
    optimize.py         objective 1 (LP, HiGHS) and 3 (QP, Clarabel); tangent and piecewise models
    study.py            baseline -> model -> optimise -> AC validation -> metrics
    io_pandapower.py    safe pandapower JSON import, pandapower script export
    templates.py        CIGRE MV example and default load profiles
    weather.py          PV / wind models and the synthetic example year
    api.py              JSON entry points used by the browser worker
  web/src/              page.html (markup + CSS), app.js (UI), worker.js
  web/vendor/pyodide/   minimal Pyodide 314.0.7 (numpy, scipy, highspy, clarabel)
  build.py              builds site/ from the sources
  tests/                validation against pandapower, lightsim2grid and the original study code
```

Rebuild after editing: copy `site/vendor` to `source/web/vendor` once, then run `python build.py` (writes `dist/`; copy it to `site/`). The tests expect the original study code in `../original` (or set ORIGINAL_CODE).

## Validation

* `tests/test_network.py`: load flow vs. the original lightsim2grid code on CIGRE MV
  (≤ 2·10⁻⁷ %-points) and vs. pandapower on 40 random networks with taps, phase shift,
  iron losses, shunts, open switches and parallel elements (≤ 1.4·10⁻⁶).
* `tests/test_vs_original.py`: optimised and AC-validated peak loadings equal those of
  `run_single_bus_scenario` for all three models and both objectives.
* `tests/test_io.py`: pandapower JSON import and script export round-trip.
