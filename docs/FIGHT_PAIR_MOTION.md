# Pair/motion pilot

This is an experiment, not an accepted replacement or a promise of camera accuracy.
The two local iPhone recordings are diagnostic evidence only and must not be uploaded
or used for training/calibration. Do not stage real impacts to improve a detector.

## Hypothesis and controls

The installed RGB-aux pilot missed close interactions despite visible people and healthy
frame delivery. Lowering its threshold alone did not recover the second recording.
The new residual head adds directed wrist-to-opponent-torso offsets, relative velocity
using sparse observation times, pair quality masks, and low-resolution optical flow.
Global median flow subtraction compensates translation, not arbitrary handheld motion.
Missing/occluded poses are masked; presence of two people is not itself an alarm.

`rgb_control` and `pair_motion` share frozen source RGB features, seed 751, source splits,
balanced sampling, and training policy. Each epoch samples 512 exposures with replacement;
these are not 512 new videos. AIRTLab camera pairs remain grouped by performance.
Only train excludes background-focused negatives under the existing documented policy.
No RGB/pose-misaligned affine augmentations are introduced. The source weight SHA-256
must match cached features; no additional backbone training is claimed.

Development selects the model; independent calibration selects its threshold. Confirmation
is fixed at 0.4 seconds before diagnostic evaluation. Public test was inspected in earlier
experiments, so results are repeated diagnostics, not a new unseen benchmark. Each variant
gets at most four minutes and 16 epochs, with early stopping on protected development
progress. The Colab notebook has a 23-minute allocation watchdog to stay below the
authorized 25-minute GPU cap including setup; it releases the runtime after completion.

## Artifacts and acceptance

Run `colab/fight_pair_motion_drive.ipynb` against existing
`SENTINEL/artifacts/fight_fusion_v1/window_cache.zip` and the matching large-v1 weights.
Outputs go to `SENTINEL/artifacts/fight_pair_motion_v1`, separate from production.
The exported schema is `rgb_pose_motion`: 16 RGB frames at 12 fps, eight sparse poses,
and float32 flow `[regions,2,15,32,32]`. Runtime and CPU QA use the same preprocessing.

Before deployment, require protected development progress over the matched RGB control,
GPU/CPU score agreement, public whole-video hard-negative checks, CPU upload/replay
diagnostics, and local-only review of missed interactions. An improvement in the two
known phone clips is not evidence of general camera accuracy. Do not tune on those clips
or silently replace a model that increases false alarms. Keep thresholds and failed runs.

## Method references

- [Two-stream convolutional networks](https://arxiv.org/abs/1406.2199) motivates complementary appearance/motion inputs; it does not validate this implementation.
- [Official Colab runtime API](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py) exposes `runtime.unassign()` for releasing the allocated runtime.
