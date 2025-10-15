# utils/show_first_values.py
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Any

import numpy as np


DEFAULT_KEYS: List[str] = [
    "x1", "y1", "z1", "u1", "phi1",
    "x2", "y2", "z2", "u2", "phi2",
]


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "results").is_dir():
            return p
    return start.parent


def default_pickle_path(start: Path) -> Path:
    root = find_repo_root(start)
    return root / "results" / "PINN Data" / "data_for_PIN.pkl"


def to_numpy(a: Any) -> np.ndarray:
    # Gracefully handle jax, torch, lists, etc.
    try:
        import jax.numpy as jnp  # type: ignore
        if isinstance(a, jnp.ndarray):
            a = np.array(a)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(a, np.ndarray):
        return a
    if isinstance(a, (list, tuple)):
        try:
            return np.asarray(a)
        except Exception:
            return np.array(a, dtype=object)
    # Last resort: wrap scalars
    return np.asarray(a)


def summarize_sample(sample: Any) -> str:
    arr = to_numpy(sample)
    if arr.ndim == 0:
        return _fmt(arr.item())
    if arr.ndim >= 1:
        flat = arr.ravel()
        n = min(6, flat.size)
        head = ", ".join(_fmt(x) for x in flat[:n])
        suffix = " …" if flat.size > n else ""
        return f"[{head}{suffix}] shape={tuple(arr.shape)}"
    return repr(sample)


def _fmt(x: Any) -> str:
    if isinstance(x, (float, np.floating)):
        return f"{x:.6g}"
    return str(x)


def print_first_values(data: dict, keys: Iterable[str], idx: int) -> int:
    width = max(len(k) for k in keys) if keys else 3
    status = 0
    for k in keys:
        if k not in data:
            print(f"{k:<{width}} : (missing)")
            status = 1
            continue
        series = data[k]
        arr = to_numpy(series)
        if arr.ndim == 0:
            # scalar stored (unusual for a time series), just print it
            print(f"{k:<{width}} : {summarize_sample(arr)}")
            continue
        if arr.shape[0] <= idx:
            print(f"{k:<{width}} : Index {idx} out of range (length={arr.shape[0]})")
            status = 2
            continue
        print(f"{k:<{width}} : {summarize_sample(arr[idx])}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show the first (or Nth) value of selected time series from data_for_PIN.pkl"
    )
    parser.add_argument(
        "-f", "--file", type=Path, default=None,
        help="Path to the pickle (defaults to (root)/results/PINN Data/data_for_PIN.pkl).",
    )
    parser.add_argument(
        "-k", "--keys", nargs="*", default=DEFAULT_KEYS,
        help=f"Keys to inspect (default: {', '.join(DEFAULT_KEYS)}).",
    )
    parser.add_argument(
        "-i", "--index", type=int, default=0,
        help="Time index to print (default: 0).",
    )
    args = parser.parse_args(argv)

    start = Path(__file__).resolve()
    pkl_path = args.file or default_pickle_path(start)

    if not pkl_path.exists():
        sys.stderr.write(f"❌ File not found: {pkl_path}\n")
        return 1

    try:
        with pkl_path.open("rb") as fh:
            data = pickle.load(fh)
    except Exception as e:
        sys.stderr.write(f"❌ Failed to load pickle: {e}\n")
        return 2

    if not isinstance(data, dict):
        sys.stderr.write(f"⚠️ Loaded object is {type(data).__name__}, not a dict—attempting best effort.\n")
        # Some users store a list with a single dict
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
            data = data[0]
        else:
            sys.stderr.write("❌ Cannot interpret structure—expected dict-like.\n")
            return 3

    print(f"📦 Loaded: {pkl_path}")
    print(f"🧭 Index: {args.index}")
    return print_first_values(data, args.keys, args.index)


if __name__ == "__main__":
    raise SystemExit(main())
