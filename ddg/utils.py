"""
ddg/utils.py
============
Shared utilities: logging setup that tees stdout/stderr to a file.
"""
import os, sys


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def start_logging(run_dir: str, fname: str = "train_log.txt") -> str:
    """Redirect stdout and stderr to both the terminal and a log file."""
    log_path = os.path.join(run_dir, fname)
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path
