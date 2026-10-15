"""The sha16 content-hash convention.

``sha16`` is the 16-hex content hash that liveness.json uses for its
provenance fields (``circuit_sha256``, ``discovered_sha256``).
"""

from __future__ import annotations

import hashlib


def sha16(data) -> str:
    """Return the 16-hex sha256 prefix of str or bytes."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]
