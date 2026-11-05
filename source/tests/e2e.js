// End-to-end browser test: load page, wait for engine, run one placement, run all, screenshots.
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const DIST = path.join(__dirname, "..", "dist");
const UPLOT = "/home/claude/wasm/node_modules/uplot/dist/uPlot.iife.min.js";
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".zip": "application/zip", ".json": "application/json", ".whl": "application/zip" };
const srv = http.createServer((req, res) => {
  const p = path.join(DIST, decodeURIComponent(req.url.split("?")[0]).replace(/^\/$/, "/index.html"));
  fs.readFile(p, (e, d) => { if (e) { res.writeHead(404); res.end(); return; } res.writeHead(200, { "Content-Type": types[path.extname(p)] || "application/octet-stream" }); res.end(d); });
}).listen(8811);
(async () => {
  const mode = process.argv[2] || "light";
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 950 }, colorScheme: mode });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  await page.route("https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js", (r) => r.fulfill({ body: fs.readFileSync(UPLOT), contentType: "text/javascript" }));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  const t0 = Date.now();
  await page.goto("http://localhost:8811/index.html");
  await page.waitForFunction(() => document.querySelector("#engine-txt").textContent.includes("ready") || document.querySelector("#engine .dot.err"), null, { timeout: 120000 });
  console.log("engine:", await page.textContent("#engine-txt"), (Date.now() - t0) / 1000, "s");
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `tests/shots/network-${mode}.png` });
  await page.click("#tab-study");
  await page.waitForTimeout(500);
  await page.screenshot({ path: `tests/shots/study-${mode}.png`, fullPage: true });
  let t = Date.now();
  await page.click("#r-run");
  await page.waitForSelector("#view-results:not([hidden]) .kpis", { timeout: 300000 }).catch(async () => { console.log("run failed:", await page.textContent("#r-msg")); });
  console.log("one placement (default model):", (Date.now() - t) / 1000, "s");
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `tests/shots/results-${mode}.png`, fullPage: true });
  if (process.argv[3] === "all") {
    await page.click("#tab-study"); await page.click("#r-all"); await page.check('input[name="model"][value="tangent"]');
    t = Date.now(); await page.click("#r-run");
    await page.waitForSelector("#view-results:not([hidden]) .rank", { timeout: 600000 }).catch(async () => console.log("all failed:", await page.textContent("#r-msg")));
    console.log("all placements (tangent):", (Date.now() - t) / 1000, "s");
    await page.waitForTimeout(800);
    await page.screenshot({ path: `tests/shots/ranking-${mode}.png`, fullPage: true });
    await page.click(".rank tbody tr:nth-child(2)");
    await page.waitForSelector("#detail .kpis", { timeout: 120000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: `tests/shots/ranking-detail-${mode}.png`, fullPage: true });
  }
  // phone width
  await page.setViewportSize({ width: 400, height: 860 }); await page.click("#tab-network"); await page.waitForTimeout(600);
  await page.screenshot({ path: `tests/shots/phone-network-${mode}.png`, fullPage: true });
  const ov = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  console.log("horizontal overflow at 400px:", ov);
  console.log("errors:", errors.length ? errors.slice(0, 10) : "none");
  await browser.close(); srv.close();
})();
