"""ddg/utils.py - automatic console logging so every run leaves a raw log."""
import sys, os, datetime


class Tee:
    """Duplicate stdout to a log file. Guarantees a raw training log exists."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout
        self.file.write(f"\n===== run started {datetime.datetime.now()} =====\n")

    def write(self, m):
        self.stdout.write(m)
        self.file.write(m)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def start_logging(out_dir, name="train_log.txt"):
    tee = Tee(os.path.join(out_dir, name))
    sys.stdout = tee
    return tee
