import copy
import hashlib
import json

import pytest

from scripts.manifest_identity import validate_pack_manifest


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("formatting", ["crlf", "compact", "key_order"])
def test_identical_annotations_survive_platform_and_json_formatting(tmp_path, formatting):
    value = {"readiness": "ready", "sources": [
        {"video": "one", "split": "train", "label": "stand_close"}]}
    reference_bytes = (json.dumps(value, indent=2) + "\n").encode()
    reference = tmp_path / "reference.json"
    reference.write_bytes(reference_bytes)
    if formatting == "crlf":
        packed = reference_bytes.replace(b"\n", b"\r\n")
    elif formatting == "compact":
        packed = json.dumps(value, separators=(",", ":")).encode()
    else:
        packed = json.dumps(value, sort_keys=True).encode()
    result = validate_pack_manifest(reference, packed, {"manifest_sha256": sha(packed)})
    assert result["formatting_differs"]
    assert result["pack_manifest_sha256"] != result["reference_manifest_sha256"]
    same = validate_pack_manifest(reference, reference_bytes, {"manifest_sha256": sha(reference_bytes)})
    assert result["manifest_content_sha256"] == same["manifest_content_sha256"]


@pytest.mark.parametrize("field,replacement", [("label", "fight"), ("split", "test"), ("video", "other")])
def test_different_labels_splits_or_media_still_fail(tmp_path, field, replacement):
    value = {"sources": [{"video": "one", "split": "train", "label": "stand_close"}]}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(value), encoding="utf-8")
    changed = copy.deepcopy(value)
    changed["sources"][0][field] = replacement
    packed = json.dumps(changed).encode()
    with pytest.raises(ValueError, match="different JSON content"):
        validate_pack_manifest(reference, packed, {"manifest_sha256": sha(packed)})


def test_corrupt_manifest_is_rejected_before_semantic_comparison(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_bytes(b'{"sources": []}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_pack_manifest(reference, reference.read_bytes(), {"manifest_sha256": "wrong"})
