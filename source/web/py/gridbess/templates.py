"""Built-in example networks (as specs)."""
import math

# CIGRE TB 575 load profiles (Fig. 6.4), share of max per hour 0-23
RESIDENTIAL = [0.26, 0.20, 0.19, 0.19, 0.19, 0.23, 0.40, 0.55, 0.62, 0.63, 0.67, 0.60,
               0.72, 0.60, 0.52, 0.49, 0.48, 0.60, 0.80, 0.90, 0.80, 0.68, 0.52, 0.38]
COMMERCIAL = [0.36, 0.34, 0.29, 0.34, 0.38, 0.47, 0.70, 0.89, 0.99, 0.97, 1.00, 0.80,
              0.82, 0.86, 0.87, 0.88, 0.80, 0.55, 0.49, 0.47, 0.42, 0.39, 0.34, 0.33]
DEFAULT_PROFILES = {"residential": RESIDENTIAL, "commercial": COMMERCIAL, "flat": [1.0] * 24}


def _c_nf(b_us_per_km, f_hz=50):
    return b_us_per_km * 1e-6 / (2 * math.pi * f_hz) * 1e9


def cigre_mv():
    """European CIGRE MV benchmark (TB 575 ch. 6.2/6.4), PV + wind, as in build_cigre_mv.py."""
    loads = [
        (1, "residential", 14.994, 3.04466), (1, "commercial", 4.845, 1.59247),
        (3, "residential", 0.27645, 0.06928), (3, "commercial", 0.22525, 0.13960),
        (4, "residential", 0.43165, 0.10818), (5, "residential", 0.72750, 0.18233),
        (6, "residential", 0.54805, 0.13735), (7, "commercial", 0.07650, 0.04741),
        (8, "residential", 0.58685, 0.14708), (9, "commercial", 0.57375, 0.35558),
        (10, "residential", 0.47530, 0.11912), (10, "commercial", 0.06800, 0.04214),
        (11, "residential", 0.32980, 0.08266), (12, "residential", 14.994, 3.04466),
        (12, "commercial", 5.016, 1.64868), (13, "commercial", 0.03400, 0.02107),
        (14, "residential", 0.20855, 0.05227), (14, "commercial", 0.33150, 0.20545),
    ]
    ug = (0.501, 0.716, 47.493)
    oh = (0.510, 0.366, 3.172)
    lines = [
        (1, 1, 2, ug, 2.82), (2, 2, 3, ug, 4.42), (3, 3, 4, ug, 0.61), (4, 4, 5, ug, 0.56),
        (5, 5, 6, ug, 1.54), (6, 6, 7, ug, 0.24), (7, 7, 8, ug, 1.67), (8, 8, 9, ug, 0.32),
        (9, 9, 10, ug, 0.77), (10, 10, 11, ug, 0.33), (11, 11, 4, ug, 0.49), (12, 3, 8, ug, 1.30),
        (13, 12, 13, oh, 4.89), (14, 13, 14, oh, 2.99), (15, 14, 8, oh, 2.00),
    ]
    # schematic coordinates (feeder 1 left, feeder 2 right), HV bus on top
    xy = {0: (4.6, 0), 1: (2.3, 1.2), 2: (2.3, 2.3), 3: (2.3, 3.4), 4: (0.0, 4.6), 5: (0.0, 5.7), 6: (0.0, 6.8),
          7: (0.0, 7.9), 8: (4.6, 4.6), 9: (4.6, 5.7), 10: (4.6, 6.8), 11: (4.6, 7.9), 12: (8.6, 1.2), 13: (8.6, 2.9),
          14: (8.6, 4.6)}
    buses = [{"id": 0, "name": "Node 0 (HV)", "vn_kv": 110.0, "x": xy[0][0], "y": xy[0][1]}]
    buses += [{"id": i, "name": f"Node {i}", "vn_kv": 20.0, "x": xy[i][0], "y": xy[i][1]} for i in range(1, 15)]
    spec = {
        "name": "CIGRE MV (TB 575) with PV and wind",
        "f_hz": 50, "sn_mva": 1.0,
        "buses": buses,
        "ext_grids": [{"bus": 0, "vm_pu": 1.02, "va_degree": 0.0}],
        "lines": [{
            "id": seg - 1, "name": f"Line {seg} ({fr}-{to})", "from_bus": fr, "to_bus": to,
            "length_km": l, "r_ohm_per_km": p[0], "x_ohm_per_km": p[1], "c_nf_per_km": _c_nf(p[2]),
            "max_i_ka": 0.283 if p is ug else 0.210, "parallel": 1, "df": 1,
            "kind": "cable" if p is ug else "overhead",
        } for seg, fr, to, p, l in lines],
        "switches": [
            {"name": "S2", "bus": 7, "line": 5, "closed": False},
            {"name": "S3", "bus": 4, "line": 10, "closed": False},
            {"name": "S1", "bus": 8, "line": 14, "closed": False},
        ],
        "trafos": [],
        "loads": [{"id": i, "name": f"Load N{n} {t.capitalize()}", "bus": n, "p_mw": p, "q_mvar": q, "profile": t}
                  for i, (n, t, p, q) in enumerate(loads)],
        "sgens": [{"id": i, "name": f"{t} N{n}", "bus": n, "p_mw": p, "q_mvar": 0.0, "type": t}
                  for i, (n, t, p) in enumerate([(3, "PV", .02), (4, "PV", .02), (5, "PV", .03), (6, "PV", .03),
                                                  (7, "WP", 1.5), (8, "PV", .03), (9, "PV", .03),
                                                  (10, "PV", .04), (11, "PV", .01)])],
        "profiles": DEFAULT_PROFILES,
    }
    z_base = 20.0 ** 2 / 25.0
    vk = math.sqrt(0.016 ** 2 + 1.92 ** 2) / z_base * 100
    vkr = 0.016 / z_base * 100
    for i, lv in enumerate((1, 12)):
        spec["trafos"].append({
            "id": i, "name": f"Trafo 0-{lv}", "hv_bus": 0, "lv_bus": lv, "sn_mva": 25.0,
            "vn_hv_kv": 110.0, "vn_lv_kv": 20.0, "vk_percent": vk, "vkr_percent": vkr,
            "pfe_kw": 0.0, "i0_percent": 0.0, "shift_degree": 30.0, "tap_side": "lv",
            "tap_pos": 0, "tap_neutral": 0, "tap_step_percent": 0.625, "parallel": 1, "df": 1,
        })
    return spec
