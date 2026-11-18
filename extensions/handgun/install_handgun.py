"""Verify and install a handgun hand-crop classifier for the runtime extension.

    extensions\env\Scripts\python.exe -m extensions.handgun.install_handgun <handgun_<RUN_TAG>.zip>

The zip comes from the handgun Colab notebook (train_hand_classifier.py --stage export with
--name handgun): handgun_spec.json ("method": "hand_crop"), the classifier and the pose weights.
Same checks and atomic install as install_sharp_object.py, into weights/extensions/handgun/.
"""

import argparse
from pathlib import Path

from extensions.sharp_object.install_sharp_object import install


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    install("handgun", archive=args.archive, hand_crop_only=True)


if __name__ == "__main__":
    main()
