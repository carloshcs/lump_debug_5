# -*- coding: utf-8 -*-
"""
material_db.py
Load materials from /materials_db/*.txt
Supports multi-line table blocks:
    [k_table]
    20, 0.25
    100, 0.27
    ...
"""

import os
import re

MATERIALS = {}

def parse_table_block(lines, start_index):
    """
    Parse a block like:
        [k_table]
        20, 0.25
        100, 0.27
    Returns (table_dict, next_index)
    """
    T, V = [], []
    i = start_index + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("[") and line.endswith("]"):
            break  # end of this block
        parts = re.split(r"[, \t]+", line)
        if len(parts) >= 2:
            try:
                T.append(float(parts[0]))
                V.append(float(parts[1]))
            except ValueError:
                pass
        i += 1
    return (T, V), i


def load_material_file(path):
    mat = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        # Table blocks
        if line.startswith("[") and line.endswith("]"):
            key = line.strip("[]").strip()
            table, i = parse_table_block(lines, i)
            mat[key] = table
            continue

        # Simple key=value pairs
        if "=" in line:
            key, val = [x.strip() for x in line.split("=", 1)]
            if key in ("rho", "k", "cp", "emissivity"):
                try:
                    mat[key] = float(val)
                except ValueError:
                    pass
            elif key == "enable_radiation":
                mat[key] = val.lower() in ("true", "1", "yes", "on")
            elif key == "name":
                mat["name"] = val
            else:
                mat[key] = val
        i += 1

    return mat


def load_all_materials(folder="materials_db"):
    global MATERIALS
    MATERIALS.clear()
    if not os.path.isdir(folder):
        print(f"[material_db] Warning: folder '{folder}' not found.")
        return MATERIALS

    for fn in os.listdir(folder):
        if fn.lower().endswith(".txt"):
            path = os.path.join(folder, fn)
            mat = load_material_file(path)
            if "name" not in mat:
                mat["name"] = os.path.splitext(fn)[0]
            MATERIALS[mat["name"]] = mat

    print(f"[material_db] Loaded {len(MATERIALS)} materials from '{folder}'")
    return MATERIALS


# Auto-load
load_all_materials()

if __name__ == "__main__":
    from pprint import pprint
    pprint(MATERIALS)
