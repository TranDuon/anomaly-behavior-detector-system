import zipfile

import cv2
import numpy as np

from scripts.fight_source_health import check_case


def test_source_health_checks_integrity_and_sequential_seek_agreement(tmp_path):
    video = tmp_path / "videos/fight/example.avi"
    video.parent.mkdir(parents=True)
    annotation = tmp_path / "annotation/example.csv"
    annotation.parent.mkdir()
    annotation.write_text("0\n" * 20)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(video, "UBI_FIGHTS/videos/fight/example.avi")
        archive.write(annotation, "UBI_FIGHTS/annotation/example.csv")
    with zipfile.ZipFile(archive_path) as archive:
        result = check_case(archive, tmp_path, "example", 0.5)
        assert result["archive_video_crc_matches"]
        assert result["archive_annotation_matches"]
        assert result["seek_equals_sequential"]
        annotation.write_text("1\n" * 20)
        assert not check_case(archive, tmp_path, "example", 0.5)[
            "archive_annotation_matches"
        ]
