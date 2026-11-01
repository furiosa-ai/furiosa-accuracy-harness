"""Run untrusted, model-written Python with limits.

Each call runs `python -I <file>` in a fresh temporary directory:
- macOS: under sandbox-exec with network denied and file writes allowed only inside that directory (and /dev);
- everywhere: CPU-time, file-size and (Linux) address-space rlimits, a wall-clock timeout, no inherited environment
  secrets (the API key is not passed through).
This keeps honest mistakes (infinite loops, huge outputs, stray file writes) contained; it is not a hardened jail, so run
the harness on a machine you are fine exposing to model-written code, never as root.
"""
import asyncio
import os
import platform
import resource
import shutil
import sys
import tempfile

PY = sys.executable
SANDBOX_EXEC = shutil.which("sandbox-exec") if platform.system() == "Darwin" else None

_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (subpath "/private{tmp}") (subpath "/dev"))
"""


def _limits(cpu_s):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        if platform.system() == "Linux":
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        os.setsid()
    return f


_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}


async def run_python(code, stdin_text, timeout_s=6.0):
    """-> (status, stdout, stderr) with status in ok | timeout | error (nonzero exit)."""
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="fah-"))
    try:
        path = os.path.join(tmp, "main.py")
        with open(path, "w") as f:
            f.write(code)
        cmd = [PY, "-I", path]
        if SANDBOX_EXEC:
            cmd = [SANDBOX_EXEC, "-p", _PROFILE.format(tmp=tmp.removeprefix("/private")), *cmd]
        p = await asyncio.create_subprocess_exec(*cmd, cwd=tmp, env={**_ENV, "HOME": tmp, "TMPDIR": tmp},
                                                 stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE, preexec_fn=_limits(int(timeout_s) + 1))
        try:
            out, err = await asyncio.wait_for(p.communicate(stdin_text.encode()), timeout_s)
        except asyncio.TimeoutError:
            try:
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass
            await p.wait()
            return "timeout", "", ""
        status = "ok" if p.returncode == 0 else "error"
        return status, out.decode(errors="replace"), err.decode(errors="replace")[-2000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
