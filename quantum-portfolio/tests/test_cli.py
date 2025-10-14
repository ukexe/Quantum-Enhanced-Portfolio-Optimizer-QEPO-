import os
import subprocess
import sys
from pathlib import Path


def test_cli_module_runs_help():
    env = os.environ.copy()
    # Ensure src layout is importable for subprocess
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (
        str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-m", "qepo.cli", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
