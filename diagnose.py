"""
Run this FIRST before main.py.
It will print exactly what filenames look like and how subjects are being grouped.
Paste the output here so we can fix main.py.
"""

import os
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = "./SOCOFing"   # same as main.py

real_dir = Path(DATASET_ROOT) / "Real"

print("=" * 60)
print("DIAGNOSTIC — SOCOFing Dataset Inspector")
print("=" * 60)

if not real_dir.exists():
    print(f"\n[ERROR] Folder not found: {real_dir.resolve()}")
    print("  Fix: set DATASET_ROOT to the folder that CONTAINS Real/ and Altered/")
    exit()

all_files = sorted([f for f in real_dir.iterdir()
                    if f.suffix.lower() in (".bmp", ".png", ".jpg", ".tif", ".tiff")])

print(f"\nTotal images in Real/: {len(all_files)}")
print(f"\nFirst 10 filenames:")
for f in all_files[:10]:
    print(f"  {f.name}")

print(f"\nLast 5 filenames:")
for f in all_files[-5:]:
    print(f"  {f.name}")

# Try splitting by different separators
print("\n--- Separator Analysis (first 5 files) ---")
for f in all_files[:5]:
    name = f.stem  # no extension
    print(f"\n  File: {f.name}")
    print(f"    Split by '__' : {name.split('__')}")
    print(f"    Split by '_'  : {name.split('_')}")
    print(f"    Split by '-'  : {name.split('-')}")
    print(f"    First char(s) before first '_': {name.split('_')[0]}")

# Group by the FIRST part before '__'
groups_dunder = defaultdict(list)
groups_single = defaultdict(list)
for f in all_files:
    name = f.stem
    groups_dunder[name.split("__")[0]].append(f.name)
    groups_single[name.split("_")[0]].append(f.name)

print(f"\n--- Grouping by split('__')[0] ---")
print(f"  Unique subjects found: {len(groups_dunder)}")
sizes = sorted(set(len(v) for v in groups_dunder.values()))
print(f"  Images-per-subject range: {sizes[:5]} ... {sizes[-5:]}")
sample_keys = list(groups_dunder.keys())[:3]
for k in sample_keys:
    print(f"  Subject '{k}': {groups_dunder[k][:3]}")

print(f"\n--- Grouping by split('_')[0] ---")
print(f"  Unique subjects found: {len(groups_single)}")
sizes = sorted(set(len(v) for v in groups_single.values()))
print(f"  Images-per-subject range: {sizes[:5]} ... {sizes[-5:]}")
sample_keys = list(groups_single.keys())[:3]
for k in sample_keys:
    print(f"  Subject '{k}': {groups_single[k][:3]}")

print("\n" + "=" * 60)
print("Paste ALL of this output as a reply so we can fix the code.")
print("=" * 60)
