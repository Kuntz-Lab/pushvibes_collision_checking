#!/usr/bin/env python3
"""Inspect the pickle files from Britton's data."""

import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/PushVIB3S_procedure_level_plan_example")


def inspect_value(val, indent=0):
    prefix = "  " * indent
    if isinstance(val, np.ndarray):
        print(f"{prefix}ndarray shape={val.shape} dtype={val.dtype} min={val.min():.4f} max={val.max():.4f}")
    elif isinstance(val, dict):
        print(f"{prefix}dict with {len(val)} keys: {list(val.keys())}")
        for k, v in val.items():
            print(f"{prefix}  [{k!r}]:")
            inspect_value(v, indent + 2)
    elif isinstance(val, (list, tuple)):
        print(f"{prefix}{type(val).__name__} len={len(val)}")
        if len(val) > 0:
            print(f"{prefix}  [0]:")
            inspect_value(val[0], indent + 2)
            if len(val) > 1:
                print(f"{prefix}  [1]:")
                inspect_value(val[1], indent + 2)
    else:
        print(f"{prefix}{type(val).__name__}: {val!r}")


def load_and_print(path):
    print(f"\n{'='*60}")
    print(f"FILE: {path.name}")
    print("=" * 60)
    with open(path, "rb") as f:
        data = pickle.load(f)
    inspect_value(data)


# Inspect goal file
load_and_print(DATA_DIR / "thanks_joe_goal.pickle")

# Inspect each vibes file
for v in [1, 2, 3]:
    load_and_print(DATA_DIR / f"thanks_joe_vibes_v{v}.pickle")
