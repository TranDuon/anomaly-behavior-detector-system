from scripts.build_mmaction_indoor_manifest import build


def source(video, dataset="airtlab", action="normal"):
    return {
        "video": video,
        "dataset": dataset,
        "split": "train",
        "group": video,
        "duration": 2,
        "intervals": [{"start_seconds": 0, "end_seconds": 2, "action": action}],
    }


def test_builder_keeps_indoor_airtlab_and_rejects_cctv_and_label_conflict():
    corpus = {
        "sources": [
            source("airtlab/non-violent/cam1/1.mp4"),
            source("airtlab/non-violent/cam1/22.mp4", action="fight"),
            source("rlvs/V_1", dataset="rlvs", action="fight"),
        ]
    }
    manifest, quarantine = build(corpus)
    assert [item["video"] for item in manifest["sources"]] == [
        "airtlab/non-violent/cam1/1.mp4"
    ]
    assert manifest["sources"][0]["environment"] == "indoor"
    assert manifest["readiness"] == "seed_only"
    reasons = {item["video"]: item["reasons"] for item in quarantine}
    assert "fight_inside_non_fight_source" in reasons[
        "airtlab/non-violent/cam1/22.mp4"
    ]
    assert "dataset_not_indoor_profile" in reasons["rlvs/V_1"]
