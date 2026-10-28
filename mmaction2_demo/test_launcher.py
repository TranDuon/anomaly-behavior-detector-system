import importlib.util
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).with_name("launch.py")
    spec = importlib.util.spec_from_file_location("isolated_mmaction_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_source_and_output_are_separate_arguments():
    module = launcher()
    source = r"D:\demo videos\standing fight.mp4"
    output = Path("result with spaces.mp4")
    args = module.command(source, output, show=False)
    assert args[args.index("--input-video") + 1] == source
    assert args[args.index("--out-filename") + 1] == str(output)
    assert "--show" not in args
    assert "--action-score-thr" in args
    assert args[args.index("--action-score-thr") + 1] == "0.4"


def test_camera_command_does_not_open_device_when_constructed():
    module = launcher()
    args = module.command("2", Path("camera.mp4"))
    assert args[args.index("--input-video") + 1] == "2"
    assert "--show" in args
    assert args[args.index("--device") + 1] == "cuda:0"
    assert str(module.ROOT / "weights" / "action_slowonly_r101.pth") in args


def test_download_rejects_unapproved_host_and_outside_destination(tmp_path):
    path = Path(__file__).with_name("download_file.py")
    spec = importlib.util.spec_from_file_location("isolated_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="official"):
        module.download("https://example.com/weights", module.ROOT / "weights/test")
    with pytest.raises(ValueError, match="inside"):
        module.download("https://download.pytorch.org/weights", tmp_path / "test")
