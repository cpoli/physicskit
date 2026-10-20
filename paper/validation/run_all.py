"""
Run every validation of the paper and write ``paper/results/validation.json``.

- ``validate_qutip.py``, ``validate_astropy.py`` and ``validate_scipy.py``:
  cross-checks against the pinned QuTiP, Astropy and SciPy (see
  ``paper/environment.yml``).
- The gallery examples of ``analytic_catalogue.ANALYTIC``, re-run here: each
  asserts its closed-form result.
- The listings of the paper (``paper/listings/*.py``), which assert theirs.

Usage (from the repository root)::

    python paper/validation/run_all.py
"""

import contextlib
import importlib.metadata as metadata
import io
import json
import platform
import runpy
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from analytic_catalogue import ANALYTIC  # noqa: E402

PACKAGES = ("physicskit", "qutip", "astropy", "numpy", "scipy", "numba")


def run(path):
    with contextlib.redirect_stdout(io.StringIO()) as out, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # plt.show() under Agg, unrendered animations
        namespace = runpy.run_path(str(path), run_name="__main__")
    plt.close("all")
    return namespace, out.getvalue()


def main():
    results = {
        "versions": {p: metadata.version(p) for p in PACKAGES},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cross_checks": {},
        "analytic": [],
    }
    for name in ("validate_qutip.py", "validate_astropy.py", "validate_scipy.py"):
        namespace, out = run(HERE / name)
        print(out, end="")
        results["cross_checks"][name[len("validate_") : -3]] = namespace["RESULTS"]
    for entry in ANALYTIC:
        start = time.perf_counter()
        run(ROOT / "examples" / entry["example"])
        seconds = time.perf_counter() - start
        results["analytic"].append({**entry, "passed": True, "seconds": round(seconds, 2)})
        print(f"passed  {entry['example']:70s} {seconds:6.1f} s")
    for listing in sorted((ROOT / "paper" / "listings").glob("*.py")):
        run(listing)
        print(f"passed  {listing.relative_to(ROOT)}")
    out = ROOT / "paper" / "results" / "validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
