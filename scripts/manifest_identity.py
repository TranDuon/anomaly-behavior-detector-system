"""Verify packaged manifest bytes, then compare JSON content across platforms."""

import hashlib
import json
from pathlib import Path


def validate_pack_manifest(reference, packed_bytes, metadata):
    """Allow whitespace/key order changes; reject corrupt packs or changed labels.

    PACKAGE.json hashes the manifest as it was packaged, which can contain CRLF
    on Windows. Git archives normalize that file to LF. Hashing the Git copy
    against the package's byte hash incorrectly rejects identical annotations.
    """
    packed_sha = hashlib.sha256(packed_bytes).hexdigest()
    if packed_sha != metadata.get("manifest_sha256"):
        raise ValueError(
            "Pack manifest checksum mismatch: manifest.json does not match PACKAGE.json"
        )
    reference_bytes = Path(reference).read_bytes()
    reference_value = json.loads(reference_bytes)
    packed_value = json.loads(packed_bytes)
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    reference_content, packed_content = canonical(reference_value), canonical(packed_value)
    if reference_content != packed_content:
        raise ValueError(
            "Pack and source manifests have different JSON content; "
            "labels/media/splits may differ. Use the matching data pack."
        )
    return {
        "pack_manifest_sha256": packed_sha,
        "reference_manifest_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "manifest_content_sha256": hashlib.sha256(reference_content).hexdigest(),
        "formatting_differs": reference_bytes != packed_bytes,
    }
