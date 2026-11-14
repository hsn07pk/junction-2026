"""Joint-space motion layer for the SO-101 follower.

Drives the real driver classes of the pinned physicalai SO-101 plugin
(physicalai_so101_plugin.SO101, plugin 0.3.0 from physicalai rev
8381d439c2660d357b4b9d633f28ea7e7ed42c81). Joint values use the driver's
"normalized" unit by default: body joints in [-100, 100], gripper in [0, 100].

Limits: no move exceeds HARD_MAX_SPEED (50 units/s) or DEFAULT_MAX_ACCEL (50 units/s^2).
The driver's own estimate (about 279 units/s for the calibrated follower) is not a safe
operating speed, so it only lowers the limit. Both values are tuning values to confirm on hardware.

Callers open a serial port only inside a claim_devices() block. The block takes the same
host-local device lock the Studio backend takes and refuses a port that another process of
this user has open (scan of /proc/<pid>/fd). The device lock alone only sees Studio when both
use the same XDG_RUNTIME_DIR; the /proc scan does not depend on that, but it cannot see
processes of other users. While a Studio runtime session or observation stream holds the arm,
claim_devices() raises DeviceOwnedError instead of sharing the port. To hand the arm over,
stop the Studio runtime session first.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger
from physicalai_so101_plugin.constants import SO101_JOINT_ORDER

if TYPE_CHECKING:
    from physicalai_so101_plugin import SO101
    from physicalai_so101_plugin.calibration import SO101Calibration

JOINT_ORDER: tuple[str, ...] = tuple(SO101_JOINT_ORDER)
GRIPPER_INDEX = JOINT_ORDER.index("gripper")
BODY_LIMIT = 100.0
GRIPPER_LOW, GRIPPER_HIGH = 0.0, 100.0
DEFAULT_BAUDRATE = 1_000_000  # so101.py:133 (STS3215 factory default)
DEFAULT_MAX_SPEED = 25.0  # normalized units per second, all joints synchronized; tune on hardware
HARD_MAX_SPEED = 50.0  # ceiling for every speed (CLI, YAML, dry run and player); no override flag
DEFAULT_MAX_ACCEL = 50.0  # normalized units per second squared, peak of the smoothstep; tune on hardware
DEFAULT_RATE_HZ = 30.0
DEFAULT_TOLERANCE = 3.0  # normalized units allowed between goal and final read-back
DEFAULT_SETTLE_S = 0.5
FILE_UNIT = "normalized"
# Binding port rule (coordinator, 13:42Z): the follower is this by-id link. The Studio export's
# follower connection_string (/dev/ttyACM1) is stale; that node is now the leader and is ignored.
FOLLOWER_PORT_BY_ID = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00"
FOLLOWER_SERIAL = "5B3D040939"

_CLAIMED_IN_PROCESS: set[str] = set()


class MotionError(ValueError):
    """Invalid pose, sequence, calibration or motion request. Nothing was sent to the arm."""


class DeviceOwnedError(RuntimeError):
    """Another process (usually Studio) already owns the serial device."""


class MotionStalledError(MotionError):
    """A step did not reach its pose and allow_stall is false. Later steps were not sent.

    results holds the MoveResult of every step that ran before this one, plus this step.
    """

    def __init__(self, message: str, results: list[MoveResult]) -> None:
        super().__init__(message)
        self.results = results


@dataclass(frozen=True)
class Step:
    """One entry of a named sequence in the poses file."""

    pose: str
    speed: float | None = None
    dwell_s: float = 0.0
    allow_stall: bool = False
    """True for steps that are expected to stop short, such as closing the gripper on a brick."""


@dataclass(frozen=True)
class MoveResult:
    """Outcome of one move_to() call, measured by reading the arm back after settling."""

    setpoints: int
    duration_s: float
    max_error: float
    arrived: bool


@dataclass(frozen=True)
class StudioRobot:
    """Connection and calibration of one robot entry from a Studio project export."""

    name: str
    port: str
    unit: str
    calibration: SO101Calibration
    serial_number: str | None = None


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def check_speed(value: float | str, label: str) -> float:
    """Return value as a speed in units/s, or raise MotionError unless 0 < value <= HARD_MAX_SPEED.

    Raises:
        MotionError: If value is not a number, is not finite, is not positive or is above HARD_MAX_SPEED.
    """
    try:
        speed = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{label} must be a number, got {value!r}"
        raise MotionError(msg) from error
    if not (math.isfinite(speed) and 0.0 < speed <= HARD_MAX_SPEED):
        msg = f"{label} must be above 0 and at most {HARD_MAX_SPEED:g} units/s, got {value!r}"
        raise MotionError(msg)
    return speed


def validate_target(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a float64 copy of a 6-joint target after checking shape, finiteness and limits.

    Raises:
        MotionError: If the shape is not (6,), a value is not finite, a body joint is
            outside [-100, 100], or the gripper is outside [0, 100].
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (len(JOINT_ORDER),):
        msg = f"expected {len(JOINT_ORDER)} values in order {JOINT_ORDER}, got shape {arr.shape}"
        raise MotionError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "joint values must be finite"
        raise MotionError(msg)
    body = arr[:GRIPPER_INDEX]
    if np.any(np.abs(body) > BODY_LIMIT):
        msg = f"body joints must lie in [-100, 100], got {body.tolist()}"
        raise MotionError(msg)
    if not GRIPPER_LOW <= arr[GRIPPER_INDEX] <= GRIPPER_HIGH:
        msg = f"gripper must lie in [0, 100], got {arr[GRIPPER_INDEX]}"
        raise MotionError(msg)
    return arr


def plan_ramp(
    start: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    max_speed: float,
    dt: float,
    max_accel: float = DEFAULT_MAX_ACCEL,
) -> np.ndarray:
    """Return setpoints that move every joint from start to target within the speed and acceleration limits.

    The profile is a smoothstep, 3u^2 - 2u^3 with u = t / T, so the arm starts and stops at rest.
    With delta the largest joint displacement, its peak rate is 1.5 * delta / T and its peak
    acceleration is 6 * delta / T^2. T = max(1.5 * delta / max_speed, sqrt(6 * delta / max_accel))
    keeps both limits. Setpoints are sampled every dt seconds and the last row is exactly the
    target. A zero move returns the target once.

    Returns:
        Float64 array of shape (n, 6), n >= 1.

    Raises:
        MotionError: If max_speed, max_accel or dt is not finite and positive, or the shapes differ.
    """
    if not (_finite_positive(max_speed) and _finite_positive(max_accel) and _finite_positive(dt)):
        msg = "max_speed, max_accel and dt must be finite and positive"
        raise MotionError(msg)
    s = np.asarray(start, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if s.shape != t.shape:
        msg = f"start shape {s.shape} differs from target shape {t.shape}"
        raise MotionError(msg)
    delta = float(np.max(np.abs(t - s)))
    if delta == 0.0:
        return t.reshape(1, -1).copy()
    duration = max(1.5 * delta / max_speed, math.sqrt(6.0 * delta / max_accel))
    steps = max(1, math.ceil(duration / dt))
    u = np.arange(1, steps + 1, dtype=np.float64) / steps
    ease = 3.0 * u**2 - 2.0 * u**3
    setpoints = s + np.outer(ease, t - s)
    setpoints[-1] = t
    return setpoints


def device_ids_for(port: str) -> tuple[str, ...]:
    """Return the device id of the node a port path points to, without touching hardware.

    The path is resolved first (a /dev/serial/by-id link becomes its /dev/ttyACM node), so the
    lock identity is the one the driver reports (SO101.device_ids, so101.py:225-227) and the one
    Studio locks. Resolving a link is a readlink, not an open.
    """
    from physicalai.robot.device_ids import device_id_from_serial_port

    return (device_id_from_serial_port(os.path.realpath(port)),)


def port_holder_pid(port: str) -> int | None:
    """Return the pid of another process of this user that has the serial port open, or None.

    Scans /proc/<pid>/fd for the resolved port path (Linux). The current process is skipped.
    Processes whose fd directory this user cannot read are skipped as well.
    """
    target = os.path.realpath(port)
    own = os.getpid()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        pid = int(fd_dir.parent.name)
        if pid == own:
            continue
        try:
            entries = list(fd_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if os.readlink(entry) == target:
                    return pid
            except OSError:
                continue
    return None


def port_serial_number(port: str) -> str | None:
    """Return the USB serial number pyserial lists for a port, or None when the port is not listed.

    Enumerates the serial ports of the host and does not open any of them.
    """
    from serial.tools import list_ports

    target = os.path.realpath(port)
    for info in list_ports.comports():
        if os.path.realpath(info.device) == target:
            return info.serial_number
    return None


@contextlib.contextmanager
def claim_devices(device_ids: Sequence[str], *, owner: str | None = None, ports: Sequence[str] = ()) -> Iterator[str]:
    """Hold the host-local device locks for the body of the with-block.

    Uses physicalai's private lock module, the same call the Studio backend makes
    (application/backend/src/workers/robot_worker.py:86-97). The lock is an flock on
    an open file descriptor, so it is released even if this process is killed.
    Each port in ports is also checked with port_holder_pid, so a Studio session that
    runs under another XDG_RUNTIME_DIR is refused too. Devices this process already
    holds are not locked again, so a claim nested inside another claim is allowed.

    Yields:
        The owner name used for the name lock (unique per process).

    Raises:
        DeviceOwnedError: If another process holds the name lock, a device lock or an open
            handle on one of the ports.
    """
    from physicalai.robot.transport._lock import DEVICE_KIND, LockContention, acquire_locks

    name = owner or f"so101-motion-{os.getpid()}"
    for port in ports:
        holder = port_holder_pid(port)
        if holder is not None:
            msg = (
                f"{port} is open in process {holder} (same user), most likely a Studio runtime session or "
                "observation stream. Stop it in Studio and rerun. This module never opens a port it does not own."
            )
            raise DeviceOwnedError(msg)
    ids = tuple(device_ids)
    if ids and set(ids) <= _CLAIMED_IN_PROCESS:
        yield name
        return
    try:
        locks = acquire_locks(name, ids)
    except LockContention as error:
        held = error.identity if error.kind == DEVICE_KIND else ", ".join(ids)
        msg = (
            f"{held} is held by another process, most likely a Studio runtime session or observation "
            "stream. Stop it in Studio and rerun. This module never opens a port it does not own."
        )
        raise DeviceOwnedError(msg) from error
    _CLAIMED_IN_PROCESS.update(ids)
    try:
        yield name
    finally:
        _CLAIMED_IN_PROCESS.difference_update(ids)
        locks.release_all()


def build_driver(
    port: str,
    calibration: SO101Calibration,
    *,
    role: str,
    unit: str = FILE_UNIT,
    baudrate: int = DEFAULT_BAUDRATE,
) -> SO101:
    """Construct the real SO101 driver without opening the port (mirrors studio_catalog.py:199-204).

    Imports the plugin lazily because it needs the feetech-servo-sdk (scservo_sdk) module.
    """
    from physicalai_so101_plugin import SO101

    return SO101(port=port, calibration=calibration, role=role, unit=unit, baudrate=baudrate)


def open_passive(
    port: str,
    calibration: SO101Calibration,
    *,
    unit: str = FILE_UNIT,
    baudrate: int = DEFAULT_BAUDRATE,
) -> SO101:
    """Connect with torque OFF so the arm can be moved by hand (teaching) or read only.

    connect() writes TORQUE_ENABLE=0 to every servo (so101.py:421 and _configure_servos,
    so101.py:597-607), so an arm that was held is released until hold_and_enable_torque
    runs; support the arm before calling this. connect() enables torque at its end only
    when role == "follower" (so101.py:421), so the driver is built with role "leader" here,
    which leaves torque_on_disconnect False.

    The port is checked with claim_devices (device locks and the /proc scan) before it is
    opened. That claim is released when this function returns, so a caller that keeps the arm
    open must hold claim_devices for the whole session (run_sequence.py and teach_poses.py do).

    Raises:
        DeviceOwnedError: If another process holds the port.
    """
    with claim_devices(device_ids_for(port), ports=(port,)):
        driver = build_driver(port, calibration, role="leader", unit=unit, baudrate=baudrate)
        driver.connect()
    logger.info("SO-101 passive (torque off) on {}", port)
    return driver


def open_follower_held(
    port: str,
    calibration: SO101Calibration,
    *,
    unit: str = FILE_UNIT,
    baudrate: int = DEFAULT_BAUDRATE,
) -> SO101:
    """Connect a follower so the goal equals the present pose before torque is enabled.

    A plain SO101(role="follower").connect() enables torque before any goal is written
    (so101.py:421), so a stale goal could move the arm. This sequence avoids that: connect
    passively (torque off), then hold_and_enable_torque reads the present ticks, checks them
    against the calibrated range, writes them as goal and enables torque. The port is claimed
    for the whole sequence. The claim is released when this function returns (see open_passive).

    Raises:
        DeviceOwnedError: If another process holds the port.
        MotionError: If a joint is outside its calibrated range. The port is closed and torque stays off.
        Exception: Re-raised after a best-effort disconnect. A follower whose torque was already
            enabled keeps holding its position.
    """
    with claim_devices(device_ids_for(port), ports=(port,)):
        driver = open_passive(port, calibration, unit=unit, baudrate=baudrate)
        try:
            hold_and_enable_torque(driver)
        except BaseException:
            driver.disconnect()
            raise
    logger.info("SO-101 follower holding present pose on {} (torque on)", port)
    return driver


def hold_and_enable_torque(driver: SO101) -> None:
    """Make a passively connected arm a held follower: goal = present raw ticks, then torque on.

    Used at connect (open_follower_held) and at the end of hand teaching, where the arm was
    moved by hand with torque off, so an older goal in the servo must not pull it back.
    The present ticks are read first (so101.py:724). If one joint is outside its calibrated
    range_min..range_max, MotionError is raised while the role is still leader and torque is
    still off, so disconnect() leaves the arm unpowered (so101.py:457-461). Otherwise the role
    becomes follower with torque_on_disconnect True (so101.py:256-260, 435-455), the raw ticks
    are written as goal without clipping (same two calls as _hold_position, so101.py:713-722),
    and torque is enabled (so101.py:706-711).

    Raises:
        ConnectionError: If the driver is not connected (so101.py:325-337).
        MotionError: If a joint's present tick is outside its calibrated range (torque stays off).
    """
    raw = np.asarray(driver._read_joint_positions(), dtype=np.int32)  # so101.py:724
    calibration = driver._require_calibration()  # so101.py:340
    for name, tick in zip(driver.JOINT_ORDER, raw.tolist(), strict=True):
        joint = calibration.joints[name]
        if not joint.range_min <= tick <= joint.range_max:
            msg = (
                f"{name} is at tick {tick}, outside its calibrated range {joint.range_min}..{joint.range_max}. "
                "Torque stays off: support the arm and move that joint back inside its range."
            )
            raise MotionError(msg)
    driver.role = "follower"  # so101.py:256-260
    driver.torque_on_disconnect = True  # so101.py:435-455, allowed for followers only
    driver._write_joint_positions(raw)  # so101.py:780, raw ticks, not clipped to the range
    driver.set_torque(enabled=True)  # so101.py:706-711


class MotionPlayer:
    """Move one connected SO-101 follower through joint targets with speed and acceleration limits."""

    def __init__(
        self,
        driver: SO101,
        *,
        max_speed: float = DEFAULT_MAX_SPEED,
        max_accel: float = DEFAULT_MAX_ACCEL,
        rate_hz: float = DEFAULT_RATE_HZ,
        tolerance: float = DEFAULT_TOLERANCE,
        settle_s: float = DEFAULT_SETTLE_S,
    ) -> None:
        """Bind to a connected driver.

        The speed limit is the lowest of max_speed, HARD_MAX_SPEED and the driver's own
        estimate (so101.py:278-294, about 279 units/s for the calibrated follower).

        Raises:
            MotionError: If rate_hz, max_speed or max_accel is not finite and positive, or
                tolerance or settle_s is not finite and >= 0.
        """
        if not (_finite_positive(rate_hz) and _finite_positive(max_speed) and _finite_positive(max_accel)):
            msg = "rate_hz, max_speed and max_accel must be finite and positive"
            raise MotionError(msg)
        if not (math.isfinite(tolerance) and tolerance >= 0 and math.isfinite(settle_s) and settle_s >= 0):
            msg = "tolerance and settle_s must be finite and >= 0"
            raise MotionError(msg)
        self.driver = driver
        self.dt = 1.0 / rate_hz
        self.max_speed = min(float(max_speed), float(driver.max_speed), HARD_MAX_SPEED)
        self.max_accel = float(max_accel)
        self.tolerance = float(tolerance)
        self.settle_s = float(settle_s)

    def current(self) -> np.ndarray:
        """Return the present pose in the driver's unit (get_observation, so101.py:487-519)."""
        return np.asarray(self.driver.get_observation().joint_positions, dtype=np.float64)

    def hold(self) -> None:
        """Stop in place: write the present raw ticks as goal, without clipping them to the calibrated range.

        Same two driver calls as _hold_position (so101.py:713-722).
        """
        raw = np.asarray(self.driver._read_joint_positions(), dtype=np.int32)  # so101.py:724
        self.driver._write_joint_positions(raw)  # so101.py:780

    def move_to(self, target: Sequence[float] | np.ndarray, *, max_speed: float | None = None) -> MoveResult:
        """Ramp from the present pose to target within the speed and acceleration limits, then settle.

        On any exception (including KeyboardInterrupt) the arm is held at its present pose
        before the exception is re-raised.

        Raises:
            MotionError: If the target is outside the allowed limits (nothing is sent).
        """
        goal = validate_target(target)
        speed = self.max_speed if max_speed is None else min(float(max_speed), self.max_speed)
        setpoints = plan_ramp(self.current(), goal, speed, self.dt, max_accel=self.max_accel)
        settle = np.repeat(goal.reshape(1, -1), math.ceil(self.settle_s / self.dt), axis=0)
        t0 = time.monotonic()
        try:
            self._send_paced(setpoints)
            self._send_paced(settle)
        except BaseException:
            with contextlib.suppress(Exception):
                self.hold()
            raise
        final = self.current()
        max_error = float(np.max(np.abs(final - goal)))
        arrived = max_error <= self.tolerance
        if not arrived:
            logger.warning("Move ended {:.2f} units from target (tolerance {:.2f})", max_error, self.tolerance)
        return MoveResult(
            setpoints=len(setpoints) + len(settle),
            duration_s=time.monotonic() - t0,
            max_error=max_error,
            arrived=arrived,
        )

    def move_gripper(self, value: float, *, max_speed: float | None = None) -> MoveResult:
        """Change only the gripper (normalized 0 to 100) and keep the body where it is."""
        target = self.current()
        target[GRIPPER_INDEX] = value
        return self.move_to(target, max_speed=max_speed)

    def run(self, steps: Sequence[Step], poses: Mapping[str, np.ndarray]) -> list[MoveResult]:
        """Execute a sequence of named poses in order, honouring per-step speed and dwell.

        A step that ends outside the tolerance stops the sequence with MotionStalledError,
        unless the step sets allow_stall. The arm is left held where it stopped
        (disconnect() holds a follower).

        Raises:
            MotionError: If a step names a pose that is not in poses.
            MotionStalledError: If a step did not reach its pose and allow_stall is false.
        """
        results: list[MoveResult] = []
        for index, step in enumerate(steps, start=1):
            if step.pose not in poses:
                msg = f"sequence uses unknown pose {step.pose!r}"
                raise MotionError(msg)
            result = self.move_to(poses[step.pose], max_speed=step.speed)
            results.append(result)
            if not result.arrived and not step.allow_stall:
                msg = f"step {index} {step.pose!r} not reached (max error {result.max_error:.2f}); sequence stopped"
                raise MotionStalledError(msg, results)
            if step.dwell_s > 0:
                time.sleep(step.dwell_s)
        return results

    def _send_paced(self, setpoints: np.ndarray) -> None:
        """Send each setpoint, then wait for the next tick.

        After a stall the next tick is one dt after the time of the send that ended
        the stall, so the arm is not sent a burst of catch-up setpoints.
        """
        next_tick = time.monotonic()
        for setpoint in setpoints:
            self.driver.send_action(setpoint.astype(np.float32))  # so101.py:521-545
            now = time.monotonic()
            next_tick = max(next_tick, now) + self.dt
            time.sleep(max(0.0, next_tick - time.monotonic()))


def describe_plan(
    steps: Sequence[Step],
    poses: Mapping[str, np.ndarray],
    *,
    start: Sequence[float] | np.ndarray,
    max_speed: float,
    rate_hz: float = DEFAULT_RATE_HZ,
    max_accel: float = DEFAULT_MAX_ACCEL,
) -> list[dict[str, Any]]:
    """Compute the setpoint plan for a sequence without touching hardware (used by --dry-run).

    The speed of each step is capped at HARD_MAX_SPEED, the same cap MotionPlayer applies.
    Each row carries the step's pose, the speed used, the number of setpoints, the largest
    per-joint rate and the largest per-joint acceleration of the planned profile. Both must
    stay within their limits.

    Raises:
        MotionError: If a step names an unknown pose, or max_speed, max_accel or rate_hz is not
            finite and positive.
    """
    if not (_finite_positive(max_speed) and _finite_positive(max_accel) and _finite_positive(rate_hz)):
        msg = "max_speed, max_accel and rate_hz must be finite and positive"
        raise MotionError(msg)
    limit = min(float(max_speed), HARD_MAX_SPEED)
    dt = 1.0 / rate_hz
    rows: list[dict[str, Any]] = []
    current = np.asarray(start, dtype=np.float64)
    for step in steps:
        if step.pose not in poses:
            msg = f"sequence uses unknown pose {step.pose!r}"
            raise MotionError(msg)
        speed = limit if step.speed is None else min(step.speed, limit)
        goal = poses[step.pose]
        setpoints = plan_ramp(current, goal, speed, dt, max_accel=max_accel)
        path = np.vstack([current.reshape(1, -1), setpoints])
        peak_rate = float(np.max(np.abs(np.diff(path, axis=0)))) / dt
        second = np.diff(path, n=2, axis=0)
        peak_accel = float(np.max(np.abs(second))) / dt**2 if second.size else 0.0
        rows.append(
            {
                "pose": step.pose,
                "speed": round(speed, 3),
                "setpoints": int(len(setpoints)),
                "duration_s": round(len(setpoints) * dt, 3),
                "peak_rate_units_s": round(peak_rate, 3),
                "peak_accel_units_s2": round(peak_accel, 3),
                "dwell_s": step.dwell_s,
            }
        )
        current = goal
    return rows


def load_calibration(path: str | Path) -> SO101Calibration:
    """Load a LeRobot-format calibration JSON (calibration.py:56-64 validates it).

    Raises:
        MotionError: If the file is unreadable, not JSON, or fails the driver's checks.
    """
    from physicalai_so101_plugin.calibration import SO101Calibration

    try:
        return SO101Calibration.from_path(path)
    except (OSError, ValueError, TypeError) as error:
        msg = f"calibration {path}: {error}"
        raise MotionError(msg) from error


def load_studio_robot(path: str | Path, name: str) -> StudioRobot:
    """Read port, unit, serial number and calibration of one robot from a Studio project export list.

    The export is a JSON list of robots, each with "name" and "payload" holding
    "connection_string", "serial_number", "unit" and "calibration" (same shape as the project_robots.json input).

    Raises:
        MotionError: If the name matches zero or several robots, the unit is not normalized,
            or the calibration is missing.
    """
    from physicalai_so101_plugin.calibration import SO101Calibration

    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        msg = f"{path} must hold a JSON list of robots"
        raise MotionError(msg)
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == name]
    if len(matches) != 1:
        msg = f"expected exactly one robot named {name!r} in {path}, found {len(matches)}"
        raise MotionError(msg)
    payload = matches[0].get("payload") or {}
    unit = payload.get("unit", FILE_UNIT)
    if unit != FILE_UNIT:
        msg = f"robot {name!r} has unit {unit!r}; this motion layer uses {FILE_UNIT!r}"
        raise MotionError(msg)
    if not isinstance(payload.get("calibration"), dict):
        msg = f"robot {name!r} has no calibration in {path}"
        raise MotionError(msg)
    try:
        calibration = SO101Calibration.from_dict(payload["calibration"])
    except (ValueError, TypeError) as error:
        msg = f"robot {name!r} calibration in {path}: {error}"
        raise MotionError(msg) from error
    serial = payload.get("serial_number")
    return StudioRobot(
        name=name,
        port=str(payload["connection_string"]),
        unit=unit,
        calibration=calibration,
        serial_number=None if serial in (None, "") else str(serial),
    )


def follower_serial_port(port: str | None = None) -> str:
    """Return the follower's serial port: always the by-id path, never the export's connection_string.

    The export's follower connection_string is stale (it names /dev/ttyACM1, which is now the leader).
    An explicit port is accepted only when it is that same by-id path.

    Raises:
        MotionError: If an explicit port names anything else.
    """
    if port is not None and port != FOLLOWER_PORT_BY_ID:
        msg = f"the follower port is {FOLLOWER_PORT_BY_ID}; refusing {port!r}"
        raise MotionError(msg)
    return FOLLOWER_PORT_BY_ID


def resolve_arm(
    *,
    studio_export: str | Path | None,
    name: str,
    port: str | None = None,
    calibration_path: str | Path | None = None,
    follower: bool = False,
) -> tuple[str, SO101Calibration]:
    """Return (serial port, calibration) for one arm from a Studio export and/or explicit overrides.

    An explicit port or calibration file wins over the export. When the export names a serial
    number, the device on the chosen port must report that serial number, the same identity
    the Studio builder uses, so a re-plugged port cannot pass for the arm. With follower=True the
    port is always the by-id path (follower_serial_port) and the device must report FOLLOWER_SERIAL;
    the export's connection_string is ignored.

    Raises:
        MotionError: If the port or the calibration cannot be determined, or the device on the
            port reports another serial number.
    """
    calibration: SO101Calibration | None = None
    expected_serial: str | None = None
    if follower:
        port = follower_serial_port(port)
        expected_serial = FOLLOWER_SERIAL
    if studio_export is not None:
        robot = load_studio_robot(studio_export, name)
        if follower:
            if robot.port != FOLLOWER_PORT_BY_ID:
                logger.warning(
                    "{}: export connection_string {} is ignored, the follower port is {}",
                    name,
                    robot.port,
                    FOLLOWER_PORT_BY_ID,
                )
            if robot.serial_number not in (None, FOLLOWER_SERIAL):
                msg = f"{name}: export serial {robot.serial_number!r} differs from the by-id serial {FOLLOWER_SERIAL!r}"
                raise MotionError(msg)
        else:
            port = port or robot.port
            expected_serial = robot.serial_number
        calibration = robot.calibration
    if calibration_path is not None:
        calibration = load_calibration(calibration_path)
    if port is None or calibration is None:
        msg = f"{name}: pass --studio-export with that robot, or both a port and a calibration JSON"
        raise MotionError(msg)
    if expected_serial is not None:
        actual = port_serial_number(port)
        if actual != expected_serial:
            msg = (
                f"{name}: {port} reports serial number {actual!r}, the export expects {expected_serial!r}. "
                "Check --port and the cable."
            )
            raise MotionError(msg)
    return port, calibration


def load_document(path: str | Path) -> dict[str, Any]:
    """Read the poses YAML file and check its header.

    Raises:
        MotionError: If joint_order or unit does not match this module.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        msg = f"{path} must be a YAML mapping"
        raise MotionError(msg)
    if doc.get("unit", FILE_UNIT) != FILE_UNIT:
        msg = f"{path}: unit must be {FILE_UNIT!r}"
        raise MotionError(msg)
    if doc.get("joint_order", list(JOINT_ORDER)) != list(JOINT_ORDER):
        msg = f"{path}: joint_order must be {list(JOINT_ORDER)}"
        raise MotionError(msg)
    return doc


def is_placeholder(path: str | Path) -> bool:
    """True when the poses file still carries the top-level placeholder flag (values not taught).

    Raises:
        MotionError: If the file is unreadable or fails load_document's header checks.
    """
    value = load_document(path).get("placeholder", False)
    if not isinstance(value, bool):
        msg = f"{path}: placeholder must be true or false"
        raise MotionError(msg)
    return value


def load_poses(path: str | Path) -> dict[str, np.ndarray]:
    """Return every named pose as a validated (6,) array in normalized units.

    Raises:
        MotionError: If a pose is malformed or outside the limits; the message names the pose.
    """
    poses: dict[str, np.ndarray] = {}
    for pose_name, values in (load_document(path).get("poses") or {}).items():
        try:
            poses[str(pose_name)] = validate_target(values)
        except MotionError as error:
            msg = f"pose {pose_name!r}: {error}"
            raise MotionError(msg) from error
    return poses


def load_sequences(path: str | Path) -> dict[str, list[Step]]:
    """Return every named sequence as a list of Step, checked against the poses in the same file.

    Raises:
        MotionError: If a step has no pose, a speed is not within (0, HARD_MAX_SPEED], or a pose is unknown.
    """
    doc = load_document(path)
    known = set((doc.get("poses") or {}).keys())
    sequences: dict[str, list[Step]] = {}
    for seq_name, items in (doc.get("sequences") or {}).items():
        steps: list[Step] = []
        for item in items or []:
            if not isinstance(item, dict) or "pose" not in item:
                msg = f"sequence {seq_name!r}: every step needs a 'pose' key"
                raise MotionError(msg)
            pose = str(item["pose"])
            if pose not in known:
                msg = f"sequence {seq_name!r} uses unknown pose {pose!r}"
                raise MotionError(msg)
            speed = item.get("speed")
            speed_value = None if speed is None else check_speed(speed, f"sequence {seq_name!r} speed")
            allow_stall = item.get("allow_stall", False)
            if not isinstance(allow_stall, bool):
                msg = f"sequence {seq_name!r}: allow_stall must be true or false"
                raise MotionError(msg)
            steps.append(
                Step(
                    pose=pose,
                    speed=speed_value,
                    dwell_s=float(item.get("dwell_s", 0.0)),
                    allow_stall=allow_stall,
                )
            )
        sequences[str(seq_name)] = steps
    return sequences


def write_pose(path: str | Path, name: str, values: Sequence[float] | np.ndarray) -> None:
    """Add or replace one named pose in the poses file and keep its other sections.

    YAML comments are not kept when the file is rewritten.

    Raises:
        MotionError: If values fail validate_target (the file is not touched).
    """
    target = validate_target(values)
    target_path = Path(path)
    doc = load_document(target_path) if target_path.exists() else {}
    poses = dict(doc.get("poses") or {})
    poses[name] = [round(float(v), 2) for v in target]
    doc.setdefault("unit", FILE_UNIT)
    doc.setdefault("joint_order", list(JOINT_ORDER))
    doc["poses"] = poses
    ordered = {key: doc[key] for key in ("unit", "joint_order", "poses", "sequences") if key in doc}
    ordered.update({key: value for key, value in doc.items() if key not in ordered})
    target_path.write_text(yaml.safe_dump(ordered, sort_keys=False), encoding="utf-8")
