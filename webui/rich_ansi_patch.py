"""
Launcher wrapper: patches rich so it emits full ANSI codes even when stdout/stderr
is a pipe (not a real terminal). Run as:

    python rich_ansi_patch.py train.py --auto_resume -opt ...

The training script receives exactly the same sys.argv as if launched directly.
traiNNer-redux is never modified.
"""
import _thread
import io
import os
import runpy
import sys
import threading


class _FakeTTY:
    """Wraps a stream so isatty() returns True, tricking rich into ANSI output."""

    def __init__(self, stream):
        self._s = stream

    def write(self, data):
        return self._s.write(data)

    def flush(self):
        return self._s.flush()

    def isatty(self):
        return True

    def fileno(self):
        # Raising here prevents rich from trying (and failing) to enable
        # Windows VT processing on the underlying file descriptor.
        raise io.UnsupportedOperation("no real fd")

    def __getattr__(self, name):
        return getattr(self._s, name)


# Replace sys.stdout/stderr with FakeTTY wrappers so ALL libraries
# (tqdm, rich, etc.) see a real TTY rather than a pipe.
sys.stdout = _FakeTTY(sys.stdout)
sys.stderr = _FakeTTY(sys.stderr)

# Patch rich.console.Console before any training code imports it.
try:
    import rich.console as _rc

    _real_init = _rc.Console.__init__

    def _patched_init(self, *args, **kwargs):
        if "file" not in kwargs:
            import sys as _sys
            base = _sys.stderr if kwargs.get("stderr") else _sys.stdout
            kwargs["file"] = _FakeTTY(base)
        kwargs.setdefault("force_terminal", True)
        kwargs.setdefault("color_system", "truecolor")
        kwargs.setdefault("width", 220)  # wide enough to avoid wrapping log lines
        _real_init(self, *args, **kwargs)

    _rc.Console.__init__ = _patched_init
except ImportError:
    pass  # rich not installed,no-op

# ── Graceful stop via sentinel file ──────────────────────────────────────────
# The GUI server writes this file when the user clicks Stop Training.
# We watch for it in a background thread and raise KeyboardInterrupt in the
# main thread,identical to Ctrl+C, triggering traiNNer-redux's save-on-exit
# handler without touching any native (Fortran/MKL) signal handlers.
_stop_file = os.environ.get("TRAIINNER_STOP_FILE")
if _stop_file:
    def _watch_stop_file():
        import time
        while True:
            if os.path.exists(_stop_file):
                try:
                    os.remove(_stop_file)
                except OSError:
                    pass
                _thread.interrupt_main()
                return
            time.sleep(0.25)
    threading.Thread(target=_watch_stop_file, daemon=True).start()

# Remove this wrapper from argv so train.py sees a clean sys.argv.
sys.argv = sys.argv[1:]

# runpy.run_path doesn't mimic `python script.py` which adds the script's
# directory to sys.path[0]. Without this, relative imports like
# `from scripts.options...` inside train.py fail with ModuleNotFoundError.
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Execute train.py as __main__ in its own directory.
runpy.run_path(sys.argv[0], run_name="__main__")
