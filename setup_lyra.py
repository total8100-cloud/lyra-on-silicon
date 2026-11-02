"""Fetch the Lyra compiler and domain specifications used (unmodified) by every experiment.

Downloads the pinned commit of https://github.com/total8100-cloud/lyra and places its
`lyra/` package at lyra_backend/experiment/lyra/, the import path the experiment scripts use
(`from experiment.lyra.auto_compiler import ...`). Set LYRA_BACKEND to use a copy elsewhere.

Usage: python setup_lyra.py
"""
import io
import os
import shutil
import urllib.request
import zipfile

LYRA_COMMIT = "e7db29d89e6ee18b2ec5b8758ca97b574e1fe6c1"
URL = f"https://github.com/total8100-cloud/lyra/archive/{LYRA_COMMIT}.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "lyra_backend", "experiment", "lyra")

if os.path.exists(DEST):
    raise SystemExit(f"{DEST} already exists; delete it to re-fetch")
print("downloading", URL)
with urllib.request.urlopen(URL) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
prefix = f"lyra-{LYRA_COMMIT}/lyra/"
for name in z.namelist():
    if name.startswith(prefix) and not name.endswith("/"):
        out = os.path.join(DEST, name[len(prefix):])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(name) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
print("Lyra placed at", DEST)
