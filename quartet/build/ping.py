"""Smallest possible DeepSeek call, to tell a hang apart from a slow model.

Usage:
  python build/ping.py                 # flash, no thinking  -- should return in ~2s
  python build/ping.py --thinking      # pro with thinking   -- the config 03 uses
  python build/ping.py --thinking --effort low
"""
import argparse
import json
import os
import sys
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--thinking", action="store_true")
ap.add_argument("--effort", default=None,
                choices=["minimal", "low", "medium", "high", "max"])
ap.add_argument("--json", action="store_true", help="request response_format json_object")
a = ap.parse_args()

key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    sys.exit("DEEPSEEK_API_KEY not set")

model = a.model or ("deepseek-v4-pro" if a.thinking else "deepseek-v4-flash")
# The word "json" must appear literally in the prompt whenever response_format is
# json_object, otherwise DeepSeek returns 400. Every prompt in 03_decompose.py says
# "Return ONLY JSON", so they satisfy it; this one now does too.
body = {"model": model,
        "messages": [{"role": "user",
                      "content": 'Return ONLY json: {"ok": true} and nothing else.'}]}
if a.thinking:
    body["thinking"] = {"type": "enabled"}
    if a.effort:
        body["reasoning_effort"] = a.effort
else:
    body["temperature"] = 0.0          # thinking mode ignores temperature
if a.json:
    body["response_format"] = {"type": "json_object"}

print(f"POST {model}  thinking={a.thinking}  effort={a.effort}  json={a.json}")
t = time.time()
try:
    r = requests.post("https://api.deepseek.com/chat/completions", json=body, timeout=180,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"})
except requests.RequestException as exc:
    sys.exit(f"FAILED after {time.time()-t:.1f}s: {exc.__class__.__name__}: {exc}")

dt = time.time() - t
print(f"HTTP {r.status_code} in {dt:.1f}s")
if r.status_code != 200:
    print(r.text[:600])
    sys.exit(1)

d = r.json()
msg = d["choices"][0]["message"]
print("content:        ", repr(msg.get("content"))[:200])
print("reasoning_content:", repr(msg.get("reasoning_content"))[:200])
print("usage:          ", json.dumps(d.get("usage", {})))
