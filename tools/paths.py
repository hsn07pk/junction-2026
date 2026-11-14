"""Run-time roots of the sorter, derived from where this file is (no fixed install path).
APP is the repository (this file is APP/tools/paths.py); R is the workspace that holds app/, envs/, run/ and logs/
(JUNCTION_ROOT overrides it). A simulated cell sets JUNCTION_RUN_ROOT."""
import os

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("JUNCTION_ROOT") or os.path.dirname(APP)
TOOLS = os.path.join(APP, "tools")
SIM_ROOT = os.environ.get("JUNCTION_RUN_ROOT") or None
RUN_ROOT = SIM_ROOT or os.path.join(R, "run")  # live frames, flags, motion server state, heartbeat
LOG_ROOT = os.path.join(SIM_ROOT, "logs") if SIM_ROOT else os.path.join(R, "logs")  # episodes, monitor logs, e-stop images
