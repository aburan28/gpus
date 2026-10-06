#!/usr/bin/env python3
"""Validate same-process graph replay across gpuckpt checkpoint/restore.

Uses the real CUDA workload; CPU unit tests exercise orchestration separately.
Every attempt writes report.json, raw child output and checkpoint commands.
Exit 77 means hardware/build prerequisites are missing, never a passing test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import subprocess
import sys
import time


class Blocked(RuntimeError):
    pass


def expected_checksum(launches, elements):
    value = 14695981039346656037
    for i in range(elements):
        word = (17 * i + 11 + 3 * launches) & 0xffffffff
        for byte in word.to_bytes(4, "little"):
            value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def validate_event(event, kind, launches, initial=None):
    if event.get("event") != kind or event.get("launches") != launches:
        raise RuntimeError(f"unexpected graph response: {event}")
    if event.get("elements") != 4096 or event.get("graph_instantiations") != 1:
        raise RuntimeError(f"invalid workload size or graph rebuild: {event}")
    if event.get("checksum") != expected_checksum(launches, 4096):
        raise RuntimeError(f"graph checksum mismatch: {event}")
    if not event.get("graph_exec") or not event.get("device_ptr"):
        raise RuntimeError("graph/allocation identity missing")
    if initial:
        for field in ("pid", "mode", "graph_exec", "device_ptr", "device_uuid", "driver_version"):
            if event.get(field) != initial.get(field):
                raise RuntimeError(f"{field} changed across checkpoint/replay")


class Workload:
    def __init__(self, binary, mode, directory, timeout):
        self.timeout = timeout
        self.buffer = bytearray()
        self.stdout_log = (directory / "workload.stdout").open("wb")
        self.stderr_log = (directory / "workload.stderr").open("wb")
        self.commands_log = (directory / "workload.commands").open("w")
        try:
            self.process = subprocess.Popen([str(binary), mode], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.stderr_log, bufsize=0)
        except Exception:
            self.stdout_log.close(); self.stderr_log.close(); self.commands_log.close()
            raise

    def read(self):
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise RuntimeError("timed out waiting for graph workload")
                data = os.read(self.process.stdout.fileno(), 4096)
                self.stdout_log.write(data); self.stdout_log.flush()
                if not data:
                    code = self.process.wait(timeout=self.timeout)
                    if code == 77:
                        raise Blocked("graph workload requires a real CUDA driver; see workload.stderr")
                    raise RuntimeError(f"graph workload exited {code}; see workload.stderr")
                self.buffer.extend(data)
                if len(self.buffer) > 65536:
                    raise RuntimeError("oversized graph response")
        line, _, self.buffer = self.buffer.partition(b"\n")
        return json.loads(line)

    def command(self, command):
        self.commands_log.write(command + "\n"); self.commands_log.flush()
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        for handle in (self.process.stdin, self.process.stdout, self.stdout_log,
                       self.stderr_log, self.commands_log):
            handle.close()


class Checkpointer:
    def __init__(self, binary, timeout, report):
        self.binary, self.timeout, self.report = binary, timeout, report

    def run(self, *arguments):
        command = [str(self.binary), *map(str, arguments)]
        record = {"command": command}
        self.report["commands"].append(record)
        started = time.monotonic_ns()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as error:
            record.update(returncode=None, timeout=True,
                          stdout=(error.stdout or b"").decode(errors="replace"),
                          stderr=(error.stderr or b"").decode(errors="replace"))
            raise RuntimeError("checkpoint command timed out") from error
        finally:
            record["wall_ns"] = time.monotonic_ns() - started
        if result.returncode:
            raise RuntimeError(f"gpuckpt {arguments[0]} exited {result.returncode}: {result.stderr.strip()}")
        return result.stdout

    def preflight(self):
        if os.environ.get("GPUCKPT_LIBCUDA"):
            raise Blocked("GPUCKPT_LIBCUDA override is forbidden for real-driver graph validation")
        version = self.run("version")
        if "cuda-header: real\n" not in version:
            raise Blocked("rebuild gpuckpt with real CUDA headers before graph validation")

    def assert_state(self, pid, expected):
        output = self.run("state", "--pid", pid)
        if output.strip() != f"pid={pid} state={expected}":
            raise RuntimeError(f"expected {expected}, got {output.strip()}")

    def roundtrip(self, pid, repo, snapshot, parent, resume):
        args = ["snapshot", "--repo", repo, "--pid", pid, "--id", snapshot]
        if parent:
            args += ["--parent", parent]
        if resume:
            args += ["--resume"]
        output = self.run(*args)
        if f"snapshot={snapshot}" not in output.splitlines():
            raise RuntimeError("checkpoint did not commit the requested snapshot")
        self.assert_state(pid, "RUNNING" if resume else "CHECKPOINTED")
        if not resume:
            self.run("restore", "--repo", repo, "--pid", pid, "--snapshot", snapshot)
            self.assert_state(pid, "RUNNING")
        self.run("verify", "--repo", repo, "--snapshot", snapshot)


def run_case(binary, mode, directory, cycles, replays, timeout, checkpointer=None):
    directory.mkdir()
    result = {"mode": mode, "checkpointed": checkpointer is not None, "events": [], "cycles": []}
    workload = Workload(binary, mode, directory, timeout)
    try:
        initial = workload.read()
        validate_event(initial, "ready", 2)
        if initial["pid"] != workload.process.pid or initial.get("mode") != mode:
            raise RuntimeError("workload PID/mode mismatch")
        result["events"].append(initial)
        launches = 2
        repo = directory / "store"
        if checkpointer:
            checkpointer.run("init", "--repo", repo, "--chunk-size", 1048576)
        parent = None
        for cycle in range(cycles):
            # Completed graph launch and an independent device read form the barrier.
            event = workload.command("check")
            validate_event(event, "checked", launches, initial)
            result["events"].append(event)
            started = time.monotonic_ns()
            if checkpointer:
                snapshot = f"graph-{cycle:04d}"
                checkpointer.roundtrip(initial["pid"], repo, snapshot, parent, resume=cycle % 2 == 0)
                parent = snapshot
                event = workload.command("check")
                validate_event(event, "checked", launches, initial)
                result["events"].append(event)
            count = replays + cycle
            event = workload.command(f"replay {count}")
            launches += count
            validate_event(event, "replayed", launches, initial)
            result["events"].append(event)
            result["cycles"].append({"cycle": cycle, "replays": count,
                                      "roundtrip_and_replay_wall_ns": time.monotonic_ns() - started,
                                      "checksum": event["checksum"]})
        event = workload.command("quit")
        validate_event(event, "done", launches, initial)
        result["events"].append(event)
        code = workload.process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"workload cleanup failed with exit {code}")
        return result
    finally:
        workload.close()


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpuckpt", type=Path, default=root / "build/gpuckpt")
    parser.add_argument("--workload", type=Path, default=root / "build/graph_replay")
    parser.add_argument("--output", type=Path, required=True, help="new directory for raw evidence and stores")
    parser.add_argument("--cycles", type=positive, default=4)
    parser.add_argument("--replays", type=positive, default=3)
    parser.add_argument("--timeout", type=positive, default=120, help="seconds per command/response")
    parser.add_argument("--modes", choices=("capture", "explicit", "both"), default="both")
    args = parser.parse_args(argv)
    if args.replays + args.cycles - 1 > 1000000:
        parser.error("replays per cycle must be at most 1000000")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "same-process, same-GPU graph replay; no CRIU/migration",
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
              "environment": {"platform": platform.platform(), "python": sys.version},
              "commands": [], "cases": []}
    for path in (Path(__file__), root / "tests/cuda/graph_replay.c", args.gpuckpt, args.workload):
        if path.is_file():
            report.setdefault("sha256", {})[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        report["revision"] = None
    exit_code = 1
    try:
        if not args.gpuckpt.is_file() or not args.workload.is_file():
            raise Blocked("build gpuckpt and graph_replay with real CUDA headers first")
        checkpointer = Checkpointer(args.gpuckpt.resolve(), args.timeout, report)
        checkpointer.preflight()
        modes = ("capture", "explicit") if args.modes == "both" else (args.modes,)
        for mode in modes:
            control = run_case(args.workload.resolve(), mode, args.output / f"{mode}-control",
                               args.cycles, args.replays, args.timeout)
            report["cases"].append(control)
            candidate = run_case(args.workload.resolve(), mode, args.output / f"{mode}-checkpoint",
                                 args.cycles, args.replays, args.timeout, checkpointer)
            report["cases"].append(candidate)
            if [c["checksum"] for c in control["cycles"]] != [c["checksum"] for c in candidate["cycles"]]:
                raise RuntimeError("checkpointed graph differs from uninterrupted control")
        report["status"], exit_code = "passed", 0
    except Blocked as error:
        report.update(status="blocked", error=str(error)); exit_code = 77
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"graph-replay: {report['status']}; evidence: {args.output / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
