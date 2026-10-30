"""OS-backed process lease on one host; released automatically on process exit.

Not a distributed lock: run a single scheduler-enabled replica across hosts.
The lock file is intentionally retained to avoid unlink/recreate races.
"""
import hashlib
import os
from pathlib import Path
import tempfile


class SchedulerLease:
    def __init__(self, path=None):
        namespace = hashlib.sha256(str(Path(__file__).resolve().parents[2]).encode()).hexdigest()[:20]
        self.path = Path(path) if path else Path(tempfile.gettempdir()) / f"sentanalysis-scheduler-{namespace}.lock"
        self.handle = None

    def acquire(self):
        if self.handle is not None:
            return True
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self):
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
