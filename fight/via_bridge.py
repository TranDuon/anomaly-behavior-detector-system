"""Thin VIA 3 temporal-project adapter; source identities stay in the review pack."""

import copy
import time
from pathlib import PurePosixPath

from fight.temporal_labels import ACTIONS, validate_annotations

ACTION_OPTIONS = {str(index + 1): action for index, action in enumerate(ACTIONS)}


def via_project(pack):
    project = {
        "project": {
            "pid": "__VIA_PROJECT_ID__",
            "rev": "__VIA_PROJECT_REV_ID__",
            "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
            "pname": "FightWatch temporal review",
            "data_format_version": "3.1.1",
            "creator": "FightWatch VIA3 adapter",
            "created": int(time.time() * 1000),
            "vid_list": [],
        },
        "config": {
            "file": {"loc_prefix": {str(index): "" for index in range(1, 5)}},
            "ui": {
                "file_content_align": "center",
                "file_metadata_editor_visible": False,
                "spatial_metadata_editor_visible": False,
                "temporal_segment_metadata_editor_visible": True,
                "spatial_region_label_attribute_id": "",
                "gtimeline_visible_row_count": "4",
            },
        },
        "attribute": {
            "1": {
                "aname": "action",
                "anchor_id": "FILE1_Z2_XY0",
                "type": 4,
                "desc": pack["annotations"]["policy"],
                "options": dict(ACTION_OPTIONS),
                "default_option_id": "",
            }
        },
        "file": {},
        "view": {},
        "metadata": {},
    }
    sources = {row["video"]: row for row in pack["annotations"]["sources"]}
    reverse_options = {action: option for option, action in ACTION_OPTIONS.items()}
    for item in pack["items"]:
        if not item.get("media"):
            continue
        path = PurePosixPath(item["media"])
        if (
            len(path.parts) != 2
            or path.parts[0] != "media"
            or path.suffix != ".mp4"
            or ".." in path.parts
            or "\\" in item["media"]
            or ":" in item["media"]
        ):
            raise ValueError("VIA must reference prepared local preview files only")
        identifier = str(len(project["file"]) + 1)
        project["project"]["vid_list"].append(identifier)
        project["file"][identifier] = {
            "fid": identifier,
            "fname": item["id"] + ".mp4",
            "type": 4,
            "loc": 2,
            "src": item["media"],
        }
        project["view"][identifier] = {"fid_list": [identifier]}
        for index, interval in enumerate(sources[item["video"]]["intervals"]):
            project["metadata"][f"{identifier}_{index}"] = {
                "vid": identifier,
                "flg": 0,
                "z": [interval["start_seconds"], interval["end_seconds"]],
                "xy": [],
                "av": {"1": reverse_options[interval["action"]]},
            }
    return project


def import_via(project, pack, rows, reviewer, method, approve=False):
    validate_annotations(pack["annotations"], rows)
    if method not in {"human", "ai_assisted"} or not reviewer.strip():
        raise ValueError(
            "VIA import needs reviewer identity and human/ai_assisted provenance"
        )
    expected = via_project(pack)
    if (
        project.get("project", {}).get("data_format_version") != "3.1.1"
        or project.get("attribute", {}).get("1") != expected["attribute"]["1"]
    ):
        raise ValueError("Unsupported VIA project/action schema")
    if (
        not expected["file"]
        or project.get("file") != expected["file"]
        or project.get("view") != expected["view"]
        or project.get("config", {}).get("file", {}).get("loc_prefix", {}).get("2")
        != ""
    ):
        raise ValueError(
            "VIA file/view mapping changed; refuse assigning labels to the wrong source"
        )
    document = copy.deepcopy(pack["annotations"])
    sources = {row["video"]: row for row in document["sources"]}
    prepared = [item for item in pack["items"] if item.get("media")]
    mapping = {
        str(index + 1): sources[item["video"]] for index, item in enumerate(prepared)
    }
    for source in mapping.values():
        source.update(
            intervals=[],
            review_status="pending",
            reviewer=reviewer.strip(),
            review_method=method,
        )
    for metadata in project.get("metadata", {}).values():
        view = str(metadata.get("vid"))
        if view not in mapping:
            raise ValueError("Unknown VIA view")
        values, positions = metadata.get("av", {}), metadata.get("z", [])
        if "1" not in values and len(positions) != 2:
            continue
        option = str(values.get("1", ""))
        if len(positions) != 2 or metadata.get("xy") or option not in ACTION_OPTIONS:
            raise ValueError(
                "Every temporal segment needs an explicit supported action; spatial annotations are not temporal labels"
            )
        if any(type(value) not in (int, float) for value in positions):
            raise ValueError("VIA segment times must be numeric seconds")
        mapping[view]["intervals"].append(
            {
                "start_seconds": positions[0],
                "end_seconds": positions[1],
                "action": ACTION_OPTIONS[option],
            }
        )
    for source in mapping.values():
        source["intervals"].sort(key=lambda interval: interval["start_seconds"])
        if approve and source["intervals"]:
            source["review_status"] = "approved"
    validate_annotations(document, rows)
    document["scope"] = (
        "VIA3 temporal import; approval requires explicit operator assertion, not independent human-gold or camera acceptance"
    )
    return document
