# Fixed-split pilot v2

Actual run results: [FIGHT_ABLATION_V2_RESULTS.md](FIGHT_ABLATION_V2_RESULTS.md). All three branches completed in ~11 minutes; none passed the development gate.

Notebook: `colab/fight_ablation_drive.ipynb`. Source: `SENTINEL/artifacts/fight_pilot_x3d_v1/`. Outputs/cache go to new `fight_ablation_v2` directories, not over the original checkpoint.

## Comparisons

1. Interaction crops with explicit detector configuration and stable training.
2. Full-frame with the same training recipe.
3. Full-frame plus up to 32 hard negatives, scored and selected from unused windows of training videos only. Validation/test videos never enter this mining step. The third model starts from the same pretrained initialization, not from validation examples.

All branches use the original validation video/start/label identities, seed 42, 16 frames at 8 FPS, normalization and maximum epochs. Holding the temporal window fixed isolates the spatial comparison and preserves labels. Testing the original pretrained temporal recipe is deferred, not silently combined with this experiment.

Training uses effective batch size 8 through gradient accumulation, class-balanced loss, gradient clipping, cosine scheduling, backbone learning rate 1e-5 and head 1e-4 for fine-tuning. BatchNorm remains frozen. Probe and fine-tune maxima are 10 and 6 epochs; early stopping uses validation AP. This is a composite recipe comparison against v1, not proof that any single change caused an improvement.

Detector weights/confidence/NMS IoU/image size are serialized in the runtime bundle and used by the local loader. Both crop/full-frame modes use the same preprocessing implementation as runtime. Explicit IoU is 0.45. Old bundles without these fields remain loadable.

## Measurement

Reports include pooled AUC/AP, within-video AUC, precision/recall/FPR at an F1-selected threshold with recall >=0.7, metrics at threshold 0.5, best recall at precision >=0.75 and best precision at recall >=0.6. Threshold sweeps include all observed score boundaries. They are diagnostic development-validation metrics, not an independent test.

Original and new errors include timestamp, label, score, and up to six false-positive and six false-negative original/crop clips per branch. Videos are converted to H.264 for browser playback. Historical caches lack people/region metadata; these fields remain null rather than fabricated.

The shared validation has already guided method design. Any apparent gain must therefore be confirmed on a new independent split/test before a demo claim. Every exported runtime remains `approved_for_demo=false`; no automatic model promotion or replacement happens.

## Budget and reproducibility

25 minutes total script budget; each branch gets at most one third, with a notebook watchdog at total +120 seconds for shutdown. Timeouts are `inconclusive`, not success. Original baseline audit/exports also consume wall time. Completed reports are reused on rerun; changing data/recipe requires a new output/cache version. Seeds reduce variation but do not guarantee bitwise deterministic GPU kernels.
