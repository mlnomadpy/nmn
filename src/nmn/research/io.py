"""Atomic, no-overwrite publication of standalone native JSON evidence."""

import json
import os
import tempfile
from pathlib import Path


def write_json_exclusive(data, path, *, sort_keys=False):
    """Publish complete strict JSON under a new filename, or leave it absent.

    Serialize before touching disk. Stage in the destination directory, flush and
    fsync the file, then atomically hard-link it to the final name. Link creation
    refuses existing files and symlinks even if another writer wins a race.
    Filesystems without hard-link support fail explicitly; there is no non-atomic
    fallback. This is file publication, not a transaction over an evidence bundle
    or a guarantee of directory-entry durability after sudden power loss.
    """
    payload = (
        json.dumps(data, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination = Path(path)
    fd, name = tempfile.mkstemp(
        prefix=".nmn-evidence-", suffix=".tmp", dir=destination.parent
    )
    staging = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
