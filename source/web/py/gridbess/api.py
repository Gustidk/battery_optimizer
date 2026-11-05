"""Entry points used by the browser worker. Everything goes in and out as JSON text."""
import json
import math

import numpy as np

from .io_pandapower import import_pandapower_json, to_pandapower_script
from .network import Network
from .study import Study
from .templates import cigre_mv
from .weather import synthetic_weather as _synthetic

_state = {"study": None}


def _clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _out(o):
    return json.dumps(_clean(o), separators=(",", ":"))


def template(name="cigre_mv"):
    return _out(cigre_mv())


def import_pandapower(text):
    spec, warnings = import_pandapower_json(text)
    return _out({"spec": spec, "warnings": warnings})


def export_script(spec_json):
    return to_pandapower_script(json.loads(spec_json))


def validate(spec_json):
    """Build the model and run one load flow at nominal power. Returns problems, if any."""
    from . import powerflow as pf
    try:
        net = Network(json.loads(spec_json))
        P, Q = pf.injections(net, np.ones((1, len(net.loads))), np.ones((1, len(net.sgens))))
        r = pf.solve(net, P, Q)
        return _out({"ok": True, "max_loading": float(r["loading"].max()) if r["loading"].size else 0.0,
                     "vm_min": float(r["vm_pu"].min()), "vm_max": float(r["vm_pu"].max()),
                     "loading": r["loading"][0].round(3).tolist(), "vm": r["vm_pu"][0].round(5).tolist()})
    except Exception as ex:
        return _out({"ok": False, "error": str(ex)})


def synthetic_weather(lat, lon, start_doy, hours):
    ghi, wind = _synthetic(float(lat), float(lon), int(start_doy), int(hours))
    return _out({"ghi": ghi.round(1).tolist(), "wind": wind.round(2).tolist()})


def new_study(spec_json, ghi_json, wind_json, settings_json, progress=None):
    cb = (lambda stage, frac: progress(stage, float(frac))) if progress is not None else None
    st = Study(json.loads(spec_json), np.asarray(json.loads(ghi_json), float),
               np.asarray(json.loads(wind_json), float), json.loads(settings_json), cb)
    _state["study"] = st
    if cb:
        cb("Running the baseline year", 0.0)
    return _out(st.baseline_summary())


def run_bus(bus, objective, model):
    return _out(_state["study"].run_bus(int(bus), objective, model, keep_series=True))


def run_all(objective, model, buses_json="null"):
    buses = json.loads(buses_json)
    return _out(_state["study"].run_all(objective, model, buses))
