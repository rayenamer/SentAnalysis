"""Verify the documented ASGI entry point starts in a real server process."""
from pathlib import Path
import subprocess
import sys
import time


def test_uvicorn_startup(tmp_path):
    log = tmp_path / "startup.log"
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "0"],
            cwd=Path(__file__).resolve().parents[1], stdout=output, stderr=output,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                text = log.read_text(encoding="utf-8")
                if "Uvicorn running on" in text:
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            assert "Uvicorn running on" in log.read_text(encoding="utf-8"), log.read_text(encoding="utf-8")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
