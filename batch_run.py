#!/usr/bin/env python3
"""Data collection without a browser: green share x strategy x battery size -> CSV.

  python batch_run.py --greens 5,15,25 --out resultater.csv        (everything, on this computer)
  python batch_run.py --greens 5 --out del_5.csv                   (one green share)
  python batch_run.py --merge del_5.csv del_15.csv del_25.csv --out samlet.csv

Uses the same engine as the website (gridbess.zip) and the built-in DK1 2025 prices from index.html.
Needs: pip install numpy scipy highspy clarabel
"""
import argparse, csv, json, os, re, sys, tempfile, time, urllib.request, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
LAT, LON, YEAR_START = 56.15, 10.21, "2025-01-01"          # Aarhus, the same place as the website
SETTINGS = {"weight_line": 1.0, "weight_trafo": 2.0, "n_levels": 9, "eta_c": 0.95, "eta_d": 0.95, "soc_init": 0.5, "dt_h": 1.0,
            "throughput_cost": 0.0, "load_factor": 1.0, "pv_stc_w_m2": 1000.0, "pv_performance_ratio": 1.0,
            "wind_v_in": 3.0, "wind_v_rated": 12.0, "wind_v_out": 25.0, "wind_exponent": 3.0}


def load_engine():
    d = tempfile.mkdtemp(prefix="gridbess_")
    with zipfile.ZipFile(os.path.join(HERE, "gridbess.zip")) as z:
        z.extractall(d)
    sys.path.insert(0, d)
    return d


def parse_list(text):
    return [float(x) for x in re.split(r"[\s,;]+", text.strip()) if x]


def strategy(pct):
    return "Max netvenlighed" if pct >= 100 else "Max indtjening" if pct <= 0 else f"Mix {pct:g} %"


def get_weather(days, synthetic):
    n = days * 24
    if synthetic:
        from gridbess.weather import synthetic_weather
        ghi, wind = synthetic_weather(LAT, LON, 1, n)
        return [float(x) for x in ghi], [float(x) for x in wind]
    end = (datetime.fromisoformat(YEAR_START) + timedelta(days=days - 1)).date().isoformat()
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s"
           "&hourly=shortwave_radiation,wind_speed_100m&wind_speed_unit=ms&timezone=auto" % (LAT, LON, YEAR_START, end))
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                hr = json.load(r)["hourly"]
            break
        except Exception as e:                           # Open-Meteo sometimes answers 429: wait and retry
            last = e; time.sleep(20 * (attempt + 1))
    else:
        raise SystemExit(f"Could not fetch the weather from Open-Meteo: {last}")
    ghi = [float(v or 0) for v in hr["shortwave_radiation"]]; wind = [float(v or 0) for v in hr["wind_speed_100m"]]
    if len(ghi) < n - 24:
        raise SystemExit(f"Open-Meteo returned {len(ghi)} hours, {n} were needed.")
    while len(ghi) < n:
        ghi.append(ghi[-1]); wind.append(wind[-1])
    return ghi[:n], wind[:n]


def get_prices(days):
    """DK1 2025 spot prices from index.html, keyed by Danish local hour exactly as the website does."""
    src = open(os.path.join(HERE, "index.html"), encoding="utf-8").read()
    B = json.loads(re.search(r"BUILTIN_PRICES\s*=\s*(\{.*?\});", src, re.S).group(1))
    t0 = datetime.fromisoformat(B["start_utc"].replace("Z", "+00:00")); tz = ZoneInfo("Europe/Copenhagen")
    sums = {}
    for i, v in enumerate(B["values"]):
        key = (t0 + timedelta(hours=i)).astimezone(tz).strftime("%Y-%m-%dT%H")
        e = sums.setdefault(key, [0.0, 0]); e[0] += v; e[1] += 1
    start = datetime.fromisoformat(YEAR_START); out = []; last = None
    for i in range(days * 24):
        e = sums.get((start + timedelta(hours=i)).strftime("%Y-%m-%dT%H"))
        if e: last = e[0] / e[1]
        out.append(last)
    first = next(v for v in out if v is not None)
    return [first if v is None else v for v in out]


_G = {}
def init_worker(path, spec, ghi, wind, price):
    sys.path.insert(0, path)
    _G.update(spec=spec, ghi=ghi, wind=wind, price=price)


def run_cell(cell):
    from gridbess.batch import batch_cell
    prm = {"settings": SETTINGS, "mw": cell["mw"], "hours": cell["hrs"], "pct": cell["pct"], "der_factor": cell["factor"],
           "objective": "quadratic_peak", "keep_schedule": True, "screen": {"mode": cell["screen"], "weeks": 4, "days_each": 7, "levels": 5, "verify": 2}}
    for attempt in range(2):
        try:
            out = json.loads(batch_cell(_G["spec"], _G["ghi"], _G["wind"], _G["price"], json.dumps(prm), None))
            return {**cell, **out}
        except Exception as e:
            err = str(e)
    return {**cell, "error": err}


FIELDS = ["Groen_andel_pct", "DER_faktor", "Strategi", "Netprioritet_pct", "BESS_MW", "BESS_timer", "BESS_MWh", "Optimal_node", "Netvenlighed_pct", "Indtjening_DKK",
          "Indtjening_tabt_ift_max_indtjening_DKK", "Netvenlighed_tabt_ift_max_netvenlighed_procentpoint", "Max_belastning_foer_pct", "Max_belastning_efter_pct",
          "Aekvivalente_cyklusser", "Dage_simuleret", "Netprioritet_brugt_pct", "Reserveloesning"]


def to_rows(results):
    ok = [r for r in results if not r.get("error")]
    groups = {}
    for r in ok:
        groups.setdefault((r["green"], r["mw"], r["hrs"]), []).append(r)
    rows = []
    for r in sorted(ok, key=lambda r: (r["green"], -r["pct"], r["mw"], r["hrs"])):
        g = groups[(r["green"], r["mw"], r["hrs"])]
        mr, mn = max(x["indtjening"] for x in g), max(x["netvenlighed_pct"] for x in g)
        rows.append([r["green"], r["factor"], strategy(r["pct"]), r["pct"], r["mw"], r["hrs"], r["mw"] * r["hrs"], r["bus_name"], r["netvenlighed_pct"], r["indtjening"],
                     mr - r["indtjening"], mn - r["netvenlighed_pct"], r["max_loading_before_pct"], r["max_loading_after_pct"], r["cycles"], r["hours"] / 24, r.get("pct_used") if r.get("pct_used") is not None else r["pct"], "Ja" if r.get("fallback") else "Nej"])
    return rows


def write_csv(path, rows, comma):
    def f(v):
        if isinstance(v, float):
            v = round(v, 4)
            return str(v).replace(".", ",") if comma else str(v)
        return str(v)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        sep = ";" if comma else ","
        fh.write(sep.join(FIELDS) + "\r\n")
        for r in rows:
            fh.write(sep.join(f(v) for v in r) + "\r\n")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        head = fh.readline(); sep = ";" if ";" in head else ","
        names = head.strip().split(sep); out = []
        for line in fh:
            if line.strip():
                out.append(dict(zip(names, line.strip().split(sep))))
    return out


def merge(files, out, excel=None, grunddata=None, serier=()):
    files = [f for f in files if not f.endswith("_decimalpunktum.csv")]
    cells = []
    for p in files:
        for d in read_csv(p):
            num = lambda k: float(d[k].replace(",", "."))
            cells.append({"green": num("Groen_andel_pct"), "factor": num("DER_faktor"), "pct": num("Netprioritet_pct"), "mw": num("BESS_MW"), "hrs": num("BESS_timer"),
                          "bus_name": d["Optimal_node"], "netvenlighed_pct": num("Netvenlighed_pct"), "indtjening": num("Indtjening_DKK"),
                          "max_loading_before_pct": num("Max_belastning_foer_pct"), "max_loading_after_pct": num("Max_belastning_efter_pct"),
                          "cycles": num("Aekvivalente_cyklusser"), "hours": num("Dage_simuleret") * 24,
                          "pct_used": num("Netprioritet_brugt_pct") if d.get("Netprioritet_brugt_pct") else None, "fallback": d.get("Reserveloesning") == "Ja"})
    rows = to_rows(cells)
    write_csv(out, rows, True); write_csv(re.sub(r"\.csv$", "", out) + "_decimalpunktum.csv", rows, False)
    print(f"Merged {len(rows)} rows into {out}")
    if excel:
        write_excel(excel, rows, grunddata, [p for p in serier])
        print("Wrote", excel)


def hour_labels(days):
    start = datetime.fromisoformat(YEAR_START)
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(days * 24)]


def write_series(path, results, days):
    """The battery schedule (MW, positive = discharging, negative = charging) of every finished simulation, one column each."""
    ok = sorted([r for r in results if not r.get("error") and r.get("schedule_mw")], key=lambda r: (r["green"], -r["pct"], r["mw"], r["hrs"]))
    if not ok:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(["Tid"] + [f"{r['green']:g}|{r['pct']:g}|{r['mw']:g}|{r['hrs']:g}|{r['bus_name']}" for r in ok]) + "\n")
        times = hour_labels(days)
        for i in range(days * 24):
            fh.write(",".join([times[i]] + [f"{r['schedule_mw'][i]:.4f}" for r in ok]) + "\n")


def write_grunddata(path, greens, days, synthetic):
    """Weather, price, the power of every load and the PV and wind production for each green share, hour by hour."""
    path_e = load_engine()
    import numpy as np
    from gridbess.api import batch_green_share
    from gridbess.study import Study
    from gridbess.templates import cigre_mv
    spec = cigre_mv()
    ghi, wind = get_weather(days, synthetic); price = get_prices(days)
    share = json.loads(batch_green_share(json.dumps(spec), json.dumps(ghi), json.dumps(wind), json.dumps(SETTINGS)))["share_at_1"]
    cols = [("Tid", hour_labels(days)), ("Globalstraaling_W_m2", ghi), ("Vindhastighed_100m_m_s", wind), ("Elpris_DKK_MWh", price)]
    base = Study(spec, np.asarray(ghi), np.asarray(wind), dict(SETTINGS, der_factor=1.0))
    load_mw = base.load_scale * base.net.load_p                       # (hours, loads), MW
    lnames = [l.get("name", f"Load {l['id']}") for l in spec["loads"]]
    for j, nm in enumerate(lnames):
        cols.append((f"Forbrug_{nm}_MW", load_mw[:, j].tolist()))
    cols.append(("Forbrug_samlet_MW", load_mw.sum(axis=1).tolist()))
    for g in greens:
        st = Study(spec, np.asarray(ghi), np.asarray(wind), dict(SETTINGS, der_factor=g / 100 / share))
        gen = st.sgen_scale * st.net.sgen_p                          # (hours, generators), MW, with the DER factor applied
        types = [str(t).upper() for t in st.net.sgen_type]
        pv = sum((gen[:, k] for k, t in enumerate(types) if t == "PV"), np.zeros(len(ghi)))
        wp = sum((gen[:, k] for k, t in enumerate(types) if t in ("WP", "WIND")), np.zeros(len(ghi)))
        cols.append((f"PV_samlet_MW_gron{g:g}", pv.tolist())); cols.append((f"Vind_samlet_MW_gron{g:g}", wp.tolist()))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(c[0] for c in cols) + "\n")
        for i in range(days * 24):
            fh.write(",".join(str(cols[0][1][i]) if k == 0 else f"{cols[k][1][i]:.5f}" for k in range(len(cols))) + "\n")


def read_plain_csv(path):
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split(",")
        data = [line.rstrip("\n").split(",") for line in fh if line.strip()]
    return head, data


def write_excel(path, rows, grunddata, serier):
    import xlsxwriter
    wb = xlsxwriter.Workbook(path)
    hf = wb.add_format({"bold": True, "text_wrap": True, "valign": "top", "bg_color": "#E8EEF0", "border": 1})
    n2 = wb.add_format({"num_format": "0.00"}); n0 = wb.add_format({"num_format": "#,##0"}); n4 = wb.add_format({"num_format": "0.0000"})
    ws = wb.add_worksheet("Resultater")
    for c, name in enumerate(FIELDS):
        ws.write(0, c, name, hf)
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row):
            if isinstance(v, (int, float)):
                ws.write_number(r, c, v, n0 if FIELDS[c] in ("Indtjening_DKK", "Indtjening_tabt_ift_max_indtjening_DKK") else n2)
            else:
                ws.write(r, c, v)
    ws.set_row(0, 48); ws.set_column(0, len(FIELDS) - 1, 16); ws.freeze_panes(1, 0); ws.autofilter(0, 0, len(rows), len(FIELDS) - 1)
    # ---- hourly series
    wt = wb.add_worksheet("Timeserier")
    head, data = read_plain_csv(grunddata)
    series = []                                                       # (label, values)
    for p in serier:
        sh, sd = read_plain_csv(p)
        for k, key in enumerate(sh[1:], 1):
            green, pct, mw, hrs, node = key.split("|")
            pct_f = float(pct)
            strat = "Max netvenlighed" if pct_f >= 100 else "Max indtjening" if pct_f <= 0 else f"Mix {pct_f:g} %"
            series.append(((float(green), -pct_f, float(mw), float(hrs)), f"Batteri_MW | {green} % groen | {strat} | {mw} MW x {hrs} h | {node}", [row[k] for row in sd]))
    series.sort(key=lambda t: t[0])
    labels = [h for h in head] + [s[1] for s in series]
    for c, name in enumerate(labels):
        wt.write(0, c, name, hf)
    n = len(data)
    for i in range(n):
        wt.write_string(i + 1, 0, data[i][0])
        for c in range(1, len(head)):
            wt.write_number(i + 1, c, float(data[i][c]), n2)
        for k, s in enumerate(series):
            wt.write_number(i + 1, len(head) + k, float(s[2][i]), n4)
    wt.set_row(0, 80); wt.set_column(0, 0, 17); wt.set_column(1, len(labels) - 1, 15); wt.freeze_panes(1, 1)
    # ---- explanation
    we = wb.add_worksheet("Forklaring")
    lines = [
        "Dataopsamling: strategisk udbygning af BESS i distributionsnettet (CIGRE MV, TB 575).",
        "",
        "Fanen Resultater: en række pr. simulering (grøn andel x strategi x batteristørrelse) for hele 2025.",
        "  Netvenlighed_pct: fald i den vægtede sum af kvadrerede spidsbelastninger (linjer vægt 1, transformere vægt 2). Højere er mere netvenlig.",
        "  Indtjening_DKK: sum over året af elpris x batteriets effekt (afladning positiv, opladning negativ), uden slidomkostning.",
        "  Optimal_node: noden med mest netvenlighed for den strategi og størrelse.",
        "  Indtjening_tabt og Netvenlighed_tabt: forskel til den bedste strategi i samme gruppe (samme grønne andel og størrelse).",
        "",
        "Fanen Timeserier: hver time i 2025 (dansk lokal tid).",
        "  Globalstraaling og Vindhastighed: vejrdata fra Open-Meteo (Aarhus). Elpris: DK1 spotpris i DKK/MWh.",
        "  Forbrug_...: effekten af hver last i MW. PV_samlet og Vind_samlet: produktion i MW for hver grøn andel (DER-faktoren er indregnet).",
        "  Batteri_MW: batteriets lademønster. Positiv = aflader (leverer til nettet), negativ = oplader. Én kolonne pr. simulering.",
        "",
        "Grøn andel = årlig produktion fra sol og vind divideret med årligt energiforbrug i nettet. DER-faktoren er sat, så andelen rammes præcist.",
    ]
    for i, t in enumerate(lines):
        we.write(i, 0, t)
    we.set_column(0, 0, 140)
    wb.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greens", default="5,15,25"); ap.add_argument("--pcts", default="100,90,70,50,0")
    ap.add_argument("--powers", default="1,2,4"); ap.add_argument("--hours", default="1,2,4")
    ap.add_argument("--days", type=int, default=365); ap.add_argument("--screen", choices=["weeks", "year"], default="weeks")
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--out", default="resultater.csv")
    ap.add_argument("--synthetic-weather", action="store_true", help="only for tests: no internet needed")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--series", default=None, help="file for the hourly battery schedules (default: serie_<out>)")
    ap.add_argument("--grunddata", default=None, help="write weather, price, loads and PV/wind to this file and stop")
    ap.add_argument("--excel", default=None, help="with --merge: also write an Excel workbook")
    ap.add_argument("--grunddata-in", default=None); ap.add_argument("--series-in", nargs="+", default=[])
    a = ap.parse_args()
    if a.grunddata:
        write_grunddata(a.grunddata, parse_list(a.greens), a.days, a.synthetic_weather); print("Wrote", a.grunddata); return
    if a.merge:
        return merge(a.merge, a.out, a.excel, a.grunddata_in, a.series_in)
    path = load_engine()
    from gridbess.api import batch_green_share
    from gridbess.templates import cigre_mv
    spec = json.dumps(cigre_mv())
    ghi, wind = get_weather(a.days, a.synthetic_weather); price = get_prices(a.days)
    ghi_j, wind_j, price_j = json.dumps(ghi), json.dumps(wind), json.dumps(price)
    share = json.loads(batch_green_share(spec, ghi_j, wind_j, json.dumps(SETTINGS)))["share_at_1"]
    greens, pcts, mws, hrs = (parse_list(x) for x in (a.greens, a.pcts, a.powers, a.hours))
    cells = [{"green": g, "factor": g / 100 / share, "pct": p, "mw": m, "hrs": h, "screen": a.screen}
             for g in greens for p in pcts for m in mws for h in hrs]
    workers = a.workers or os.cpu_count() or 2
    print(f"Running {a.days} days of 2025 (screening: {a.screen}).", flush=True)
    print(f"{len(cells)} simulations, {workers} parallel processes. DER factors: " +
          ", ".join(f"{g:g} % -> x{g / 100 / share:.2f}" for g in greens), flush=True)
    t0 = time.time(); results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(path, spec, ghi_j, wind_j, price_j)) as ex:
        futs = [ex.submit(run_cell, c) for c in cells]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); results.append(r)
            el = time.time() - t0; eta = el / i * (len(cells) - i)
            print(f"[{i}/{len(cells)}] {r['green']:g} % green, {strategy(r['pct'])}, {r['mw']:g} MW x {r['hrs']:g} h: " +
                  (f"node {r['bus_name']}, netvenlighed {r['netvenlighed_pct']:.1f} %, indtjening {r['indtjening']:.0f}" if not r.get("error") else "ERROR " + r["error"]) +
                  f" | elapsed {el / 60:.0f} min, about {eta / 60:.0f} min left", flush=True)
            rows = to_rows(results); write_csv(a.out, rows, True)        # partial results are always on disk
    rows = to_rows(results); write_csv(a.out, rows, True); write_csv(re.sub(r"\.csv$", "", a.out) + "_decimalpunktum.csv", rows, False)
    write_series(a.series or os.path.join(os.path.dirname(a.out) or ".", "serie_" + os.path.basename(a.out)), results, a.days)
    bad = [r for r in results if r.get("error")]
    print(f"Done in {(time.time() - t0) / 60:.1f} min. {len(rows)} rows written to {a.out}" + (f", {len(bad)} failed" if bad else ""))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
