"""Windows stand-in for the part of ``fcntl`` FlashInfer uses: ``flock``.

Windows Fix: the ``fcntl`` module does not exist on Windows. Callers do
``try: import fcntl`` / ``except ImportError: from flashinfer import
_fcntl_compat as fcntl`` and keep their ``fcntl.flock`` calls unchanged.

``flock`` is emulated with a msvcrt byte-range lock on byte 0 of the file.
Shared locks are taken as exclusive (msvcrt has no shared mode), which is
safe but serializes readers. Locks are released when the handle is closed.
"""

import msvcrt
import os
import time

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(fd, operation: int) -> None:
    if not isinstance(fd, int):
        fd = fd.fileno()
    # msvcrt locks from the current position: always lock/unlock byte 0.
    os.lseek(fd, 0, os.SEEK_SET)
    if operation & LOCK_UN:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # not locked: flock(LOCK_UN) is a no-op as well
        return
    while True:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError as e:
            if operation & LOCK_NB:
                raise BlockingIOError(
                    e.errno, "file is locked by another process"
                ) from e
            time.sleep(0.05)
