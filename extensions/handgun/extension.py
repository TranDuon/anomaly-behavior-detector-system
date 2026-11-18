"""Runtime "person holding a handgun" extension (extensions/worker.py, extensions/env).

Same hand-crop method as the sharp-object extension (extensions/sharp_object/extension.py):
pose -> square crop around each visible hand -> binary ResNet "this hand holds a handgun" ->
everyday-object veto -> YOLO-World draws the gun box for the overlay. Only the per-weapon
settings differ; the pose pass is shared with sharp_object when both run in the same worker.

Why a separate extension and not a class of sharp_object: a gun is not a sharp object (its own
alert label and event), and the tuned sharp-object checkpoint (hand_04) stays untouched.

Installed by extensions/handgun/install_handgun.py into weights/extensions/handgun/ with
handgun_spec.json ("method": "hand_crop"). Until then the runtime does not enable it.
"""

from extensions.base import WEIGHTS, register
from extensions.sharp_object.extension import SharpObjectExtension

# A small gun in a hand is what a COCO detector most often calls a phone, a remote, a hair drier
# or a mouse, so those classes must NOT cancel a handgun alert (they stay in the sharp-object veto).
# They are trained against as hard negatives instead (datasets/handgun/class_map.json).
HANDGUN_VETO_CLASSES = frozenset({"bottle", "cup", "book", "wine glass", "toothbrush", "vase"})
HANDGUN_LOCATOR_CLASSES = ("handgun", "pistol", "revolver")


@register
class HandgunExtension(SharpObjectExtension):
    name = "handgun"
    display_label = "HANDGUN"
    confirm_seconds = 0.5
    release_seconds = 1.0
    threshold = 0.5
    method = "hand_crop"
    locator_classes = HANDGUN_LOCATOR_CLASSES
    veto_classes = HANDGUN_VETO_CLASSES
    use_tool_veto = False  # the mop / broom veto is for long blades only
    weapon_class = "handgun"

    def load(self, weights_dir=WEIGHTS):
        super().load(weights_dir)
        if self.method != "hand_crop":
            raise ValueError("handgun only supports the hand_crop method (handgun_spec.json)")
