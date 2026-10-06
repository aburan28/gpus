"""CPU protocol checks, not CUDA graph execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("graph_replay", Path(__file__).resolve().parents[1] / "scripts/graph-replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def event(kind="ready", launches=2):
    return dict(event=kind, launches=launches, elements=4096,
                checksum=replay.expected_checksum(launches, 4096), pid=123,
                mode="capture", device_ptr="1000", graph_exec="2000",
                device_uuid="abcd", driver_version=13040, graph_instantiations=1)


class ProtocolTests(unittest.TestCase):
    def test_valid_event(self):
        replay.validate_event(event(), "ready", 2)

    def test_bad_checksum(self):
        bad = event(); bad["checksum"] = "0" * 16
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            replay.validate_event(bad, "ready", 2)

    def test_bad_launch_count(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            replay.validate_event(event(launches=3), "ready", 2)

    def test_graph_rebuilt(self):
        bad = event(); bad["graph_instantiations"] = 2
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            replay.validate_event(bad, "ready", 2)

    def test_changed_identities(self):
        for field in ("pid", "mode", "device_ptr", "graph_exec", "device_uuid", "driver_version"):
            with self.subTest(field=field):
                bad = event("replayed", 5); bad[field] = "changed"
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    replay.validate_event(bad, "replayed", 5, event())

    def test_separate_restore_uses_current_snapshot(self):
        pointer = replay.Checkpointer(Path("gpuckpt"), 1, {"commands": []})
        outputs = ["snapshot=current\n", "pid=123 state=CHECKPOINTED\n", "restored=current\n", "pid=123 state=RUNNING\n", "current ok\n"]
        with patch.object(pointer, "run", side_effect=outputs) as run:
            pointer.roundtrip(123, Path("store"), "current", "previous", resume=False)
        self.assertEqual(run.call_args_list[2].args, ("restore", "--repo", Path("store"), "--pid", 123, "--snapshot", "current"))
        self.assertIn("--parent", run.call_args_list[0].args)

    def test_resume_does_not_restore_twice(self):
        pointer = replay.Checkpointer(Path("gpuckpt"), 1, {"commands": []})
        with patch.object(pointer, "run", side_effect=["snapshot=current\n", "pid=123 state=RUNNING\n", "ok\n"]) as run:
            pointer.roundtrip(123, Path("store"), "current", None, resume=True)
        self.assertIn("--resume", run.call_args_list[0].args)
        self.assertNotIn("restore", [c.args[0] for c in run.call_args_list])

    def test_failed_state_stops_restore(self):
        pointer = replay.Checkpointer(Path("gpuckpt"), 1, {"commands": []})
        with patch.object(pointer, "run", side_effect=["snapshot=current\n", "pid=123 state=FAILED\n"]) as run:
            with self.assertRaisesRegex(RuntimeError, "CHECKPOINTED"):
                pointer.roundtrip(123, Path("store"), "current", None, resume=False)
        self.assertEqual(run.call_count, 2)

    def test_mock_build_and_override_are_blocked(self):
        pointer = replay.Checkpointer(Path("gpuckpt"), 1, {"commands": []})
        with patch.dict(os.environ, {"GPUCKPT_LIBCUDA": "mock.so"}):
            with self.assertRaises(replay.Blocked): pointer.preflight()
        with patch.dict(os.environ, {}, clear=True), patch.object(pointer, "run", return_value="cuda-header: mock (UNVERIFIED)\n"):
            with self.assertRaises(replay.Blocked): pointer.preflight()

    def test_failed_command_retains_evidence(self):
        report = {"commands": []}
        pointer = replay.Checkpointer(Path("gpuckpt"), 1, report)
        result = subprocess.CompletedProcess([], 3, "partial\n", "target RESTORING\n")
        with patch.object(subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "exited 3"): pointer.run("restore")
        self.assertEqual(report["commands"][0]["returncode"], 3)
        self.assertEqual(report["commands"][0]["stderr"], result.stderr)

    def test_missing_prerequisites_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evidence"
            code = replay.main(["--gpuckpt", str(Path(temp) / "missing"), "--output", str(output)])
            report = json.loads((output / "report.json").read_text())
        self.assertEqual(code, 77)
        self.assertEqual(report["status"], "blocked")

    def test_partial_line_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            child = directory / "partial"
            child.write_text("#!/usr/bin/env python3\nimport time\nprint('{', end='', flush=True)\ntime.sleep(30)\n")
            child.chmod(0o755)
            workload = replay.Workload(child, "capture", directory, 0.1)
            try:
                with self.assertRaisesRegex(RuntimeError, "timed out"): workload.read()
            finally: workload.close()
            self.assertIsNotNone(workload.process.poll())


if __name__ == "__main__": unittest.main()
