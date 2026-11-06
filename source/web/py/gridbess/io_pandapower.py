"""
Read a pandapower JSON export (pp.to_json) WITHOUT pandapower and without
instantiating anything from the file: only plain tables and numbers are
decoded (the same safety idea as pypsa-gui's importer by A. Kies, MIT).

Also writes a network spec back out as a plain pandapower Python script.
"""
import json
import math

from .network import NetworkError, vector_group_shift

SUPPORTED = {"bus", "line", "trafo", "load", "sgen", "ext_grid", "switch", "shunt"}
IGNORED = {"res_", "std_types", "controller", "measurement", "group", "poly_cost", "pwl_cost",
           "bus_geodata", "line_geodata", "version", "format_version", "converged", "OPF_converged",
           "name", "f_hz", "sn_mva", "user_pf_options"}


def _table(obj):
    """pandas DataFrame stored as {"_object": "<split-oriented json>"} -> list of row dicts."""
    if not isinstance(obj, dict) or obj.get("_class") != "DataFrame":
        return None
    inner = obj.get("_object")
    if isinstance(inner, str):
        inner = json.loads(inner)
    if not isinstance(inner, dict):
        return None
    cols = inner.get("columns") or []
    idx = inner.get("index") or []
    data = inner.get("data") or []
    rows = []
    for i, r in zip(idx, data):
        d = {c: v for c, v in zip(cols, r)}
        d["_index"] = i
        rows.append(d)
    return rows


def _num(v, default=0.0):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _on(r):
    v = r.get("in_service", True)
    return True if v is None else bool(v)


def import_pandapower_json(text):
    """Returns (spec, warnings). Raises ValueError with a readable message on bad input."""
    try:
        root = json.loads(text)
    except json.JSONDecodeError as ex:
        raise ValueError(f"The file is not valid JSON ({ex.msg}, line {ex.lineno}).")
    obj = root.get("_object") if isinstance(root, dict) and "_object" in root else root
    if not isinstance(obj, dict) or "bus" not in obj:
        raise ValueError("This does not look like a pandapower network export (no 'bus' table). "
                         "Export it with pandapower.to_json(net, 'file.json').")
    warn = []
    T = {}
    for key, val in obj.items():
        tab = _table(val)
        if tab is None:
            continue
        if key in SUPPORTED:
            T[key] = tab
        elif not any(key.startswith(p) for p in IGNORED) and tab:
            active = [r for r in tab if _on(r)]
            if active:
                if key == "storage":
                    warn.append(f"Dropped {len(active)} storage unit(s): the battery is placed by this tool.")
                elif key == "gen":
                    warn.append(f"Dropped {len(active)} voltage-controlled generator(s) ('gen'): only static "
                                "generators ('sgen') are supported. Re-create them as sgen if you need them.")
                else:
                    warn.append(f"Dropped {len(active)} '{key}' element(s): this element type is not supported.")

    buses_raw = [b for b in T.get("bus", []) if _on(b)]
    if not buses_raw:
        raise ValueError("The network has no buses in service.")

    # closed bus-bus switches fuse buses (union-find)
    parent = {int(b["_index"]): int(b["_index"]) for b in buses_raw}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    switches = T.get("switch", [])
    open_trafos = set()
    line_switches = []
    for sw in switches:
        et = sw.get("et"); closed = bool(sw.get("closed", True))
        b = int(sw["bus"]); el = int(sw["element"])
        if et == "b":
            if closed and b in parent and el in parent:
                parent[find(b)] = find(el)
        elif et == "t":
            if not closed:
                open_trafos.add(el)
        elif et == "l":
            if not closed:
                line_switches.append({"bus": b, "line": el, "closed": False, "name": sw.get("name")})
        else:
            warn.append(f"Ignored a switch of type '{et}'.")
    rep = {b: find(b) for b in parent}
    n_fused = len(rep) - len(set(rep.values()))
    if n_fused:
        warn.append(f"Merged {n_fused} bus(es) joined by closed bus-bus switches.")

    buses = []
    for b in buses_raw:
        i = int(b["_index"])
        if rep[i] != i:
            continue
        d = {"id": i, "name": str(b.get("name") if b.get("name") is not None else f"Bus {i}"),
             "vn_kv": _num(b.get("vn_kv"))}
        geo = b.get("geo")
        if isinstance(geo, str) and geo:
            try:
                c = json.loads(geo).get("coordinates")
                if isinstance(c, list) and len(c) >= 2:
                    d["x"], d["y"] = float(c[0]), -float(c[1])
            except (ValueError, AttributeError, TypeError):
                pass
        buses.append(d)
    alive = {b["id"] for b in buses}

    def bus_ok(i):
        i = int(i)
        return i in rep and rep[i] in alive

    def R(i):
        return rep[int(i)]

    lines = []
    for l in T.get("line", []):
        if not _on(l) or not bus_ok(l["from_bus"]) or not bus_ok(l["to_bus"]):
            continue
        if R(l["from_bus"]) == R(l["to_bus"]):
            warn.append(f"Dropped line {l['_index']}: both ends sit on the same (merged) bus.")
            continue
        lines.append({"id": int(l["_index"]), "name": str(l.get("name") or f"Line {l['_index']}"),
                      "from_bus": R(l["from_bus"]), "to_bus": R(l["to_bus"]),
                      "length_km": _num(l.get("length_km"), 1.0), "r_ohm_per_km": _num(l.get("r_ohm_per_km")),
                      "x_ohm_per_km": _num(l.get("x_ohm_per_km")), "c_nf_per_km": _num(l.get("c_nf_per_km")),
                      "g_us_per_km": _num(l.get("g_us_per_km")), "max_i_ka": _num(l.get("max_i_ka"), 0.0),
                      "parallel": _num(l.get("parallel"), 1.0), "df": _num(l.get("df"), 1.0),
                      "kind": "overhead" if l.get("type") == "ol" else "cable"})
    line_ids = {l["id"] for l in lines}
    sw_out = []
    for sw in line_switches:
        if sw["line"] in line_ids and bus_ok(sw["bus"]):
            sw_out.append({**sw, "bus": R(sw["bus"])})

    trafos = []
    for t in T.get("trafo", []):
        if not _on(t) or int(t["_index"]) in open_trafos or not bus_ok(t["hv_bus"]) or not bus_ok(t["lv_bus"]):
            continue
        tct = t.get("tap_changer_type")
        if tct not in (None, "Ratio") and _num(t.get("tap_pos")) != _num(t.get("tap_neutral")):
            warn.append(f"Transformer {t['_index']}: tap changer type '{tct}' is treated as a ratio tap changer.")
        vg = t.get("vector_group")
        vg = vg.strip() if isinstance(vg, str) and vg.strip() else None
        if vg:
            try:
                if abs(((_num(t.get("shift_degree")) - vector_group_shift(vg)) + 180) % 360 - 180) > 1e-6:
                    warn.append(f"Transformer {t['_index']}: vector group {vg} does not match its phase shift of "
                         f"{_num(t.get('shift_degree')):g} degrees. The phase shift was kept and the vector group dropped.")
                    vg = None
            except NetworkError:
                warn.append(f"Transformer {t['_index']}: '{vg}' is not a recognised vector group and was dropped.")
                vg = None
        trafos.append({"id": int(t["_index"]), "vector_group": vg, "name": str(t.get("name") or f"Trafo {t['_index']}"),
                       "hv_bus": R(t["hv_bus"]), "lv_bus": R(t["lv_bus"]), "sn_mva": _num(t.get("sn_mva")),
                       "vn_hv_kv": _num(t.get("vn_hv_kv")), "vn_lv_kv": _num(t.get("vn_lv_kv")),
                       "vk_percent": _num(t.get("vk_percent")), "vkr_percent": _num(t.get("vkr_percent")),
                       "pfe_kw": _num(t.get("pfe_kw")), "i0_percent": _num(t.get("i0_percent")),
                       "shift_degree": _num(t.get("shift_degree")), "tap_side": t.get("tap_side"),
                       "tap_pos": _num(t.get("tap_pos")), "tap_neutral": _num(t.get("tap_neutral")),
                       "tap_step_percent": _num(t.get("tap_step_percent")),
                       "parallel": _num(t.get("parallel"), 1.0), "df": _num(t.get("df"), 1.0)})

    loads = []
    zip_warned = False
    for l in T.get("load", []):
        if not _on(l) or not bus_ok(l["bus"]):
            continue
        sc = _num(l.get("scaling"), 1.0)
        nm = str(l.get("name") or f"Load {l['_index']}")
        low = nm.lower()
        prof = "commercial" if ("commercial" in low or "industr" in low) else "residential"
        if not zip_warned and any(_num(l.get(k)) for k in ("const_z_p_percent", "const_i_p_percent",
                                                          "const_z_percent", "const_i_percent")):
            warn.append("Some loads are voltage dependent (ZIP); they are modelled as constant power here.")
            zip_warned = True
        loads.append({"id": int(l["_index"]), "name": nm, "bus": R(l["bus"]), "p_mw": _num(l.get("p_mw")) * sc,
                      "q_mvar": _num(l.get("q_mvar")) * sc, "profile": prof})
    sgens = []
    unknown = 0
    for s in T.get("sgen", []):
        if not _on(s) or not bus_ok(s["bus"]):
            continue
        sc = _num(s.get("scaling"), 1.0)
        typ = str(s.get("type") or "").upper(); nm = str(s.get("name") or f"Sgen {s['_index']}")
        txt = (typ + " " + nm.upper())
        if "PV" in txt or "SOLAR" in txt:
            kind = "PV"
        elif "WP" in txt or "WIND" in txt:
            kind = "WP"
        else:
            kind = "CONST"; unknown += 1
        sgens.append({"id": int(s["_index"]), "name": nm, "bus": R(s["bus"]), "p_mw": _num(s.get("p_mw")) * sc,
                      "q_mvar": _num(s.get("q_mvar")) * sc, "type": kind})
    if unknown:
        warn.append(f"{unknown} static generator(s) had no PV or wind type and were set to constant output. Check them under Generation.")
    shunts = [{"bus": R(s["bus"]), "p_mw": _num(s.get("p_mw")), "q_mvar": _num(s.get("q_mvar")),
               "step": _num(s.get("step"), 1.0)}
              for s in T.get("shunt", []) if _on(s) and bus_ok(s["bus"])]
    ext = [{"bus": R(e["bus"]), "vm_pu": _num(e.get("vm_pu"), 1.0), "va_degree": _num(e.get("va_degree"))}
           for e in T.get("ext_grid", []) if _on(e) and bus_ok(e["bus"])]
    if not ext:
        warn.append("The file has no external grid. Add one before running.")
    if any(l["max_i_ka"] <= 0 for l in lines):
        warn.append("Some lines have no thermal rating (max_i_ka). Set it before running.")
    spec = {"name": str(obj.get("name") or "Imported pandapower network"), "f_hz": _num(obj.get("f_hz"), 50.0),
            "sn_mva": _num(obj.get("sn_mva"), 1.0), "buses": buses, "lines": lines, "trafos": trafos,
            "switches": sw_out, "loads": loads, "sgens": sgens, "shunts": shunts, "ext_grids": ext}
    return spec, warn


def to_pandapower_script(spec):
    """A plain Python script that rebuilds the network with pandapower."""
    L = ["import pandapower as pp", "",
         f"net = pp.create_empty_network(name={spec.get('name', 'network')!r}, f_hz={spec.get('f_hz', 50)}, sn_mva={spec.get('sn_mva', 1.0)})",
         "", "# buses"]
    for b in spec["buses"]:
        L.append(f"pp.create_bus(net, vn_kv={b['vn_kv']}, name={b.get('name', '')!r}, index={b['id']}"
                 + (f", geodata=({b['x']}, {-b['y']})" if "x" in b and "y" in b else "") + ")")
    L.append("\n# external grid")
    for e in spec["ext_grids"]:
        L.append(f"pp.create_ext_grid(net, bus={e['bus']}, vm_pu={e.get('vm_pu', 1.0)}, va_degree={e.get('va_degree', 0)})")
    L.append("\n# lines")
    for l in spec["lines"]:
        L.append(f"pp.create_line_from_parameters(net, from_bus={l['from_bus']}, to_bus={l['to_bus']}, "
                 f"length_km={l['length_km']}, r_ohm_per_km={l['r_ohm_per_km']}, x_ohm_per_km={l['x_ohm_per_km']}, "
                 f"c_nf_per_km={l.get('c_nf_per_km', 0)}, max_i_ka={l['max_i_ka']}, g_us_per_km={l.get('g_us_per_km', 0)}, "
                 f"parallel={l.get('parallel', 1)}, df={l.get('df', 1)}, name={l.get('name', '')!r}, index={l['id']})")
    if spec.get("switches"):
        L.append("\n# line switches")
        for s in spec["switches"]:
            L.append(f"pp.create_switch(net, bus={s['bus']}, element={s['line']}, et='l', closed={bool(s.get('closed', True))}"
                     + (f", name={s['name']!r}" if s.get("name") else "") + ")")
    if spec.get("trafos"):
        L.append("\n# transformers")
        for t in spec["trafos"]:
            tap = ""
            if t.get("tap_side"):
                tap = (f", tap_side={t['tap_side']!r}, tap_pos={t.get('tap_pos', 0)}, tap_neutral={t.get('tap_neutral', 0)}, "
                       f"tap_step_percent={t.get('tap_step_percent', 0)}, tap_changer_type='Ratio'")
            L.append(f"pp.create_transformer_from_parameters(net, hv_bus={t['hv_bus']}, lv_bus={t['lv_bus']}, sn_mva={t['sn_mva']}, "
                     f"vn_hv_kv={t['vn_hv_kv']}, vn_lv_kv={t['vn_lv_kv']}, vkr_percent={t['vkr_percent']}, vk_percent={t['vk_percent']}, "
                     f"pfe_kw={t.get('pfe_kw', 0)}, i0_percent={t.get('i0_percent', 0)}, shift_degree={t.get('shift_degree', 0)}, "
                     + (f"vector_group={t['vector_group']!r}, " if t.get("vector_group") else "")
                     + f"parallel={t.get('parallel', 1)}, name={t.get('name', '')!r}, index={t['id']}{tap})")
    if spec.get("shunts"):
        L.append("\n# shunts")
        for s in spec["shunts"]:
            L.append(f"pp.create_shunt(net, bus={s['bus']}, q_mvar={s['q_mvar']}, p_mw={s.get('p_mw', 0)}, step={s.get('step', 1)})")
    L.append("\n# loads (profile name in the comment)")
    for l in spec["loads"]:
        L.append(f"pp.create_load(net, bus={l['bus']}, p_mw={l['p_mw']}, q_mvar={l.get('q_mvar', 0)}, name={l.get('name', '')!r})  # {l.get('profile', '')}")
    L.append("\n# static generators")
    for s in spec["sgens"]:
        L.append(f"pp.create_sgen(net, bus={s['bus']}, p_mw={s['p_mw']}, q_mvar={s.get('q_mvar', 0)}, name={s.get('name', '')!r}, type={s.get('type', 'PV')!r})")
    L += ["", "pp.runpp(net)", "print(net.res_line.loading_percent.max())", "# pp.to_json(net, 'network.json')"]
    return "\n".join(L) + "\n"
