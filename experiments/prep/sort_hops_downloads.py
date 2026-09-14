#!/usr/bin/env python3
"""
sort_hops_downloads.py -- move the hops_*.json files downloaded from Drive (outputs/scan/<dataset>/) into
results/qualitative/drive_scan_<dataset>/, identifying each file by its exact byte size (Drive renames duplicates to
"hops_frame_ccmp (1).json", so the name alone does not say which dataset a file belongs to).

    python3 prep/sort_hops_downloads.py [--src ~/Downloads] [--map hops_files.json]
The map is {dataset: {filename: [drive_id, size_bytes]}}.
"""
import argparse, glob, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DEFAULT_MAP = {
    "mir": {"hops_frame_ccmp.json": 2031048, "hops_frame_ccmp_off.json": 1986760, "hops_openie.json": 1144545},
    "sir4_matsci": {"hops_frame_ccmp.json": 8230898, "hops_frame_ccmp_off.json": 8150976, "hops_openie.json": 4702111},
    "sir4_biology": {"hops_frame_ccmp.json": 10090720, "hops_frame_ccmp_off.json": 11360281, "hops_openie.json": 4787163},
    "sir4_physics": {"hops_frame_ccmp.json": 8975729, "hops_frame_ccmp_off.json": 10434305, "hops_openie.json": 4494008},
    "tomato": {"hops_frame_ccmp.json": 5135287, "hops_frame_ccmp_off.json": 5123506, "hops_t4_openie.json": 357339},
    "sir4_cs": {"hops_t4_openie.json": 353223},
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=os.path.expanduser("~/Downloads")); ap.add_argument("--map", default=None); a = ap.parse_args()
    m = DEFAULT_MAP
    if a.map:
        m = {ds: {fn: v[1] if isinstance(v, list) else v for fn, v in files.items()} for ds, files in json.load(open(a.map)).items()}
    by_size = {size: (ds, fn) for ds, files in m.items() for fn, size in files.items()}
    moved, unknown = [], []
    for p in sorted(glob.glob(f"{a.src}/hops_*.json")):
        size = os.path.getsize(p)
        if size not in by_size: unknown.append((os.path.basename(p), size)); continue
        ds, fn = by_size[size]; dst = f"{ROOT}/results/qualitative/drive_scan_{ds}/{fn}"
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(p, dst); moved.append((ds, fn, size))
        json.load(open(dst))                                   # must parse
    for ds, fn, size in moved: print(f"  {ds:14s} {fn:26s} {size:>10,d} B")
    for name, size in unknown: print(f"  ? not in the map: {name} ({size:,d} B)")
    missing = [(ds, fn) for ds, files in m.items() for fn in files if not os.path.exists(f"{ROOT}/results/qualitative/drive_scan_{ds}/{fn}")]
    print(f"moved {len(moved)}; still missing: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
