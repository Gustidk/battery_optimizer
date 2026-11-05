import { loadPyodide } from "../web/vendor/pyodide/pyodide.mjs";
import fs from "node:fs";
const t0 = Date.now();
const py = await loadPyodide({ indexURL: new URL("../web/vendor/pyodide/", import.meta.url).pathname });
await py.loadPackage(["numpy", "scipy", "highspy", "clarabel"]);
py.unpackArchive(new Uint8Array(fs.readFileSync(new URL("../web/gridbess.zip", import.meta.url))), "zip", { extractDir: "/home/pyodide" });
py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
const api = py.pyimport("gridbess.api");
console.log(`startup ${(Date.now()-t0)/1000}s`);
const spec = api.template();
const hours = Number(process.argv[2] || 8760);
const w = JSON.parse(api.synthetic_weather(56.15, 10.21, 1, hours));
let t = Date.now();
const prog = (s, f) => {};
const sum = JSON.parse(api.new_study(spec, JSON.stringify(w.ghi), JSON.stringify(w.wind), "{}", prog));
console.log(`baseline ${hours}h: ${(Date.now()-t)/1000}s, peak loading ${sum.loading.flat().reduce((a,b)=>Math.max(a,b),0).toFixed(2)}`);
for (const [model, obj] of [["tangent","minmax"],["tangent","quadratic_peak"],["tangent_hourly","minmax"],["tangent_hourly","quadratic_peak"],["piecewise","minmax"],["piecewise","quadratic_peak"]]) {
  t = Date.now();
  const r = JSON.parse(api.run_bus(8, obj, model));
  console.log(`${model} ${obj}: ${(Date.now()-t)/1000}s  mean_of_max ${r.mean_of_max_before_pct.toFixed(3)} -> ${r.mean_of_max_after_pct.toFixed(3)}  timing ${JSON.stringify(r.timing)}`);
}
