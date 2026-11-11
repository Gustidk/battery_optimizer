// Runs the Python engine (gridbess) in Pyodide, off the main thread.
// Protocol: {id, cmd, args} -> {id, ok, result | error}; progress: {type:"progress", id, stage, frac}
import { loadPyodide } from "./vendor/pyodide/pyodide.mjs";

let api = null;
let current = 0;

async function boot() {
  const base = new URL("./vendor/pyodide/", import.meta.url).href;
  const py = await loadPyodide({ indexURL: base });
  postMessage({ type: "boot", stage: "Loading numpy, scipy and solvers" });
  await py.loadPackage(["numpy", "scipy", "highspy", "clarabel"]);
  postMessage({ type: "boot", stage: "Loading the optimisation engine" });
  const zip = await (await fetch(new URL("./gridbess.zip" + new URL(import.meta.url).search, import.meta.url))).arrayBuffer();
  py.unpackArchive(new Uint8Array(zip), "zip", { extractDir: "/home/pyodide" });
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  api = py.pyimport("gridbess.api");
  postMessage({ type: "ready" });
}
const ready = boot().catch((e) => postMessage({ type: "fatal", error: String(e && e.message || e) }));

function progress(stage, frac) {
  postMessage({ type: "progress", id: current, stage: String(stage), frac: Number(frac) });
}

const handlers = {
  template: () => api.template(),
  import_pp: ([text]) => api.import_pandapower(text),
  export_script: ([spec]) => api.export_script(spec),
  validate: ([spec]) => api.validate(spec),
  synthetic: ([lat, lon, doy, hours]) => api.synthetic_weather(lat, lon, doy, hours),
  new_study: ([spec, ghi, wind, settings, price]) => api.new_study(spec, ghi, wind, settings, progress, price || undefined),
  run_bus: ([bus, obj, model]) => api.run_bus(bus, obj, model),
  compare_bus: ([bus, obj, model]) => api.compare_bus(bus, obj, model),
  run_all: ([obj, model, buses]) => api.run_all(obj, model, buses),
  loadflow: () => api.loadflow(),
  hour_detail: ([hour, bus, p]) => api.hour_detail(hour, bus == null ? -1 : bus, p || 0),
};

onmessage = async (ev) => {
  const { id, cmd, args } = ev.data;
  await ready;
  if (!api) { postMessage({ id, ok: false, error: "The engine could not start." }); return; }
  current = id;
  try {
    const out = handlers[cmd](args || []);
    postMessage({ id, ok: true, result: out });
  } catch (e) {
    let msg = String(e && e.message || e);
    const m = msg.match(/(\w*Error): ([^\n]+)\s*$/);
    if (m) msg = m[2];
    postMessage({ id, ok: false, error: msg });
  }
};
