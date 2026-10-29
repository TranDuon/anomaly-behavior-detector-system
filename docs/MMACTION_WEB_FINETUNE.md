# MMAction2 web and interaction fine-tuning

The CPU FightWatch runtime remains the default. The web selector adds an isolated GPU MMAction2 original runtime and an optional trained interaction branch. Python 3.13/CPU web code never imports MMCV; file-backed local IPC calls the separate Python 3.10/CUDA environment. Both uploaded files and live camera samples use this same runtime, timestamp sampling and event gate. Workers close after processing. Original AVA labels are displayed per person, and recordings include action predictions.

## Training scope

`colab/mmaction_interaction_drive.ipynb` runs at most 45 GPU minutes including installation. It downloads only official frozen checkpoints and the existing 43 MB public review archive on Drive. There is no full AVA dataset on Drive; the initial training is NOT end-to-end AVA fine-tuning. Backbone, original AVA head and person detector remain frozen. A new order-invariant, multiple-instance interaction head is trained from per-person video features. Public temporal interval labels supervise an entire candidate bag, never every visible person. This structurally preserves original model parameters without pretending that clip labels are person annotations.

The manifest has 64 AI-assisted reviewed public sources with retained performance-group splits. It is a small, repeatedly examined diagnostic, not independent human gold or classroom/phone acceptance. No private recordings are uploaded, trained on or used for thresholds. Unknown time is not normal. Negative windows must be fully covered by known normal intervals. Threshold uses calibration only; best head uses worst-cohort development AP; test follows frozen selection. Reports separate AIRTLab and RLVS and compare original AVA fight scores against the new head. Original thresholds are not recalibrated on test. Completed runs are not retrained; interrupted test resumes the frozen head. Feature cache fingerprints bind code, source, preprocessing and corpus.

Input audit: the AVA training decoder returns RGB; the old webcam demo normalizes OpenCV BGR without converting channels. The new shared engine explicitly uses RGB, the config's mean/std, short side 256, and eight frames at 8/30-second spacing. A ninth trailing sample supplies causal decision delay; detection uses sample four. Tests verify channel normalization and sampling shape. This is a correction against the training data path, not proof that every missed action is caused by color order.

## Use

### Synchronizing another computer

Pull `main` and restart the idle web server. No weights changed in the session-log, overlay, playback or original-score calibration updates. Keep or install the same FightWatch runtime ZIP and MMAction2 official weights and interaction ZIP separately. Original-score pilot calibration is shipped as `mmaction2_demo/config/original_calibration.json`, so the calibrated selector works after pulling code without downloading another checkpoint. It is experimental: the small repeated-public test found six events and three false alerts; it is not classroom acceptance. The uncalibrated original selector remains available.

Each successful source start gets a new session ID and empty event list; old evidence files remain on disk. Person boxes change color rather than drawing another overlay rectangle. Pair-level regions are not attributed to individual victims. Realtime preview runs independently of inference, up to the source FPS capped at 30, with stale overlays suppressed. Upload defaults to realtime preview; untick realtime for complete offline analysis and annotated export. Realtime may drop model samples when overloaded and must not be confused with full-timeline offline processing.

1. Start the normal local web server at port 8000. Choose `MMAction2 gốc` and enable pilot to test the integrated original checkpoint. Selecting an option does not open a camera; the user must start a source.
2. Run the Colab notebook with GPU and connect Drive. Artifacts go to `SENTINEL/artifacts/mmaction_interaction_v1`. The notebook uses a separate Linux Python 3.10 environment and automatically releases GPU after saving progress.
3. Download `mmaction_interaction_pilot.zip`; stop web recognition, then drag the ZIP onto `mmaction2_demo/install_branch.cmd`. It verifies schema/source/checkpoint hashes, backs up a previous branch, and does not replace original AVA or FightWatch weights.
4. Choose `MMAction2 + nhánh fine-tuned` in the web and enable pilot. Missing weights produce an explicit error. Original AVA labels stay separate from the new binary event score. The new head does not add a verified separate kick-person AVA class.

The UI and recording controls are convenience, not accuracy improvements. No deployment is automatically approved. A small public score cannot guarantee classroom or iPhone performance. A proper next stage needs more independently reviewed in-domain interaction data before unfreezing the backbone or advertising an 80% acceptance result.
