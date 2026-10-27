# RGB / pose / action supervision — protocol v1

## Scope

User-authorized progress-gated training, shared CPU upload/live runtime. No promise of 80% or physical iPhone acceptance; no automatic promotion. Keep old checkpoint and exact public splits. PKU is optional diagnostic only, not required for training.

## Architecture and preprocessing

16 RGB frames at 12fps, 182px pair/context regions; official `yolov8n-pose.pt` at 320px supplies boxes and 17 COCO keypoints. Eight sparse pose observations per causal window, up to two spatially associated people per region. Coordinates normalized by pair center/body height to remove global translation and scale; absent/low-confidence joints zeroed. Association is a short-window heuristic, not proof of persistent identity under occlusion/crowds. One visible person is not automatically rejected. Pair selection is bounded and includes full-frame context; crowded-ROI coverage still requires testing.

Start from large-v1 X3D weights, frozen original RGB head, small RGB projection and temporal pose CNN. Zero-initialized residual fight head preserves initial RGB predictions. Six weak multi-label auxiliary classes: fight, hug, handshake/greet, high-five/clap, gesture, solo motion. Publisher clip tags are not verified action timestamps and may not all appear in a central window. Unknown RLVS auxiliary labels masked, not treated as negatives. Friendly-punch clips excluded as policy-ambiguous without automatic relabeling. Horizontal flips swap left/right joints; person-slot swapping prevents relying on arbitrary detector order. Mild RGB handheld augmentation is not real-phone validation.

Compare `rgb_aux` versus `rgb_pose_aux`, same crops/splits/seed; RGB-only ablation still uses pose detector for identical region preprocessing. This is not a claim of an RGB-only speed-optimized deployment. Sparse pose and bounded array-owner cache reduce repeated CPU work; actual end-to-end latency must be measured, not inferred from parameter count.

## Data / selection / progression

Use all eligible train/development/calibration entries of large-v1 manifest; test remains outside training/selection. RGB feature cache always derived from baseline weights, keyed by video SHA, position/config and actual RGB/pose weight SHA. Train sampler balances source × binary class, 256 exposures/group/epoch with replacement; counts remain unique source clips, exposures are not independent new examples.

3 feature probes + at most 18 last-two-block fine-tuning epochs, 25-minute budget per variant and 90-minute overall budget (data staging/install separate). After three fine-tuning epochs, longer training requires worst-cohort P/R improvement >=0.01 or macro AP improvement >=0.005 over the initial reprocessed development baseline, with no source AP drop >0.02. Patience 4, development ties broken by AP then BCE. Fixed seed and TF32 disabled. Best fully completed epoch always restored on budget stop. Test values never control continuation or variant ranking.

Global threshold chosen from independent calibration score-group midpoints. Confirmation 0/0.4/0.8s compared on deterministic source/class-balanced development whole-video traces; labels only support any-alarm clip evaluation, not precise event latency. Gate has no future input. Frozen configuration recorded before public test/PKU diagnostic. PKU train sample is not used at all in this run. Known public test/PKU previously inspected: repeated diagnostic, not a fresh acceptance set. Report hard-negative class counts/FP, each dataset separately, and PKU one-to-one temporal event metrics without loosening matching rules after seeing results.

## Runtime / delivery

Baseline is also evaluated with matched new pose-derived regions/12fps windows, with its own independent calibration, to avoid attributing preprocessing changes to fusion learning. Saved reports distinguish paired camera clips from unique performances. The report renderer only reads saved predictions/history and generates learning curves and timelines; it cannot tune thresholds or promote a checkpoint.

Backward-compatible `input_kind=rgb` or `rgb_pose` metadata. Fusion bundle includes TorchScript, JSON and fixed-basename official pose weights; missing pose file blocks activation rather than silently changing input. Shared pose preprocessing/model/gate applies to upload, paced replay and live sources. Existing stale-result, source-gap and insufficient-FPS protections remain. CPU tests must include model export and actual upload/replay, not only fake predictors. Do not claim iPhone acceptance until connected and measured.

Notebook: `colab/fight_fusion_drive.ipynb`. Drive run directory: `SENTINEL/artifacts/fight_fusion_v1`. Existing run is read-only, never automatic retry. Cache archive backed up after the run. Failure reason and completed checkpoint/history remain saved. Next phase only if results justify it: true multi-scale temporal branch, additional independent action sources, and physical-phone/crowd acceptance.

`scripts.fight_fusion_qa` packages both runtime ZIPs, deterministic SHA-selected hug/high-five/gesture negatives and public positives, plus optional mixed PKU source and RGB/pose tensors. It never selects cases by model scores. Unpack refuses traversal, oversized archives and nonempty output directories. `scripts.fight_interaction_cpu_smoke` supports both model input schemas; `scripts.fight_cpu_acceptance` installs into isolated candidate folders and exercises actual upload and paced camera-replay APIs. These checks do not prove real-phone or held-out event accuracy.
