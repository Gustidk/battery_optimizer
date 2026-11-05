"""Build dist/: one index.html (page + app.js + CIGRE example) plus worker.js, engine zip and Pyodide."""
import json, os, shutil, subprocess, sys, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web"); DIST = os.path.join(ROOT, "dist")
sys.path.insert(0, os.path.join(WEB, "py"))
from gridbess.templates import cigre_mv
os.makedirs(DIST, exist_ok=True)
# engine zip
zp = os.path.join(DIST, "gridbess.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(os.path.join(WEB, "py", "gridbess"))):
        if fn.endswith(".py"):
            z.write(os.path.join(WEB, "py", "gridbess", fn), f"gridbess/{fn}")
page = open(os.path.join(WEB, "src", "page.html")).read()
app = open(os.path.join(WEB, "src", "app.js")).read()
app = app.replace("/*CIGRE_JSON*/", "const CIGRE = " + json.dumps(cigre_mv(), separators=(",", ":")) + ";")
page = page.replace("/*APP_JS*/", app)
# GitHub Pages / standalone: a full document
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + page.replace("<header", "</head>\n<body>\n<header", 1) + "\n</body>\n</html>\n")
open(os.path.join(DIST, "index.html"), "w").write(full)
# artifact: body fragment (the artifact host adds the document skeleton)
open(os.path.join(DIST, "artifact.html"), "w").write(page)
shutil.copy(os.path.join(WEB, "src", "worker.js"), DIST)
vd = os.path.join(DIST, "vendor", "pyodide")
if not os.path.isdir(vd):
    shutil.copytree(os.path.join(WEB, "vendor", "pyodide"), vd)
print("built", os.path.getsize(os.path.join(DIST, "index.html")), "bytes index.html")
