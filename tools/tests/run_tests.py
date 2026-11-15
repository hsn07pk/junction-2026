"""Plain test runner for the offline replay suite (pytest is not importable in R/envs/data-env).

Run (one command, see README.md):
  nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I -B \
    app/tools/tests/run_tests.py
Exit code 0 when no test FAILs or ERRORs. SKIP is reported with its reason; a PASS may carry a note (returned string).
"""
import importlib
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import conftest as C  # noqa: E402  (also sets up sys.path for the code under test)

MODULES = ["test_perception", "test_kin", "test_loopfix", "test_safety", "test_servo_corr", "test_survey",
           "test_survey_sim", "test_defects_sim", "test_calib_wristcam", "test_underside"]


def main():
    t_all = time.time()
    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0}
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            t0 = time.time()
            try:
                note = fn()
                status, msg = "PASS", (note if isinstance(note, str) else "")
            except C.Skip as exc:
                status, msg = "SKIP", str(exc)
            except AssertionError as exc:
                status, msg = "FAIL", str(exc) or "assertion failed"
            except Exception as exc:  # noqa: BLE001
                status, msg = "ERROR", f"{type(exc).__name__}: {exc}"
            counts[status] += 1
            print(f"{status:5s} {modname}.{name} ({time.time() - t0:.2f} s) {msg}", flush=True)
    wall = time.time() - t_all
    print(f"TOTAL {sum(counts.values())} PASS {counts['PASS']} FAIL {counts['FAIL']} "
          f"ERROR {counts['ERROR']} SKIP {counts['SKIP']} wall {wall:.1f} s", flush=True)
    shutil.rmtree(C.RUN_DIR, ignore_errors=True)
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
