# Public punch/kick comparison

Run `colab/fight_punch_kick_drive.ipynb` on a GPU runtime. It fetches main without overwriting the Drive working tree, stages only the fingerprinted public sources, compares the complete installed RGB-aux classifier against official AVA Slow R50 and SlowFast R50, and saves reports in `SENTINEL/artifacts/fight_punch_kick_pretrained_v1`. No training or automatic deployment occurs. A watchdog disconnects the runtime within 45 minutes, including setup.

The corpus pins AIRTLab source revision and SHA-256 hashes. Performance groups cannot cross calibration/development/test. The threshold uses calibration only; the event gate and model use development; test runs only after selection is frozen. Baseline thresholds are recalibrated under this protocol, not presented as the installed application's original operating point. Inference uses past frames only, a 0.5-second decision cadence, and a one-second AVA window; SlowFast packs 8 slow and 32 fast frames. Timing is GPU source replay, not CPU/webcam throughput.

AVA action 64 is fight/hit person and action 71 is kick person; kick object is not used. Per-action alarm fractions measure detection within reviewed intervals, not mutually exclusive action classification. No handshake/hug veto is added: these can coexist with violence. Public selection alone cannot authorize a production checkpoint change.

Limitations: AI-assisted AIRTLab interval labels, very few independent performances, no separate development hit performance, and a previously examined single PKU recording. Mixed hit/kick intervals remain separately identified. Unlabeled gaps are excluded, not assumed normal. The two private iPhone recordings are never uploaded or used for thresholds, training, or candidate selection. Actual phone kicking and CPU deployment parity still require testing after public selection; this comparison does not guarantee all camera scenarios.

Primary references:
- [PyTorchVideo official AVA inference](https://github.com/facebookresearch/pytorchvideo/blob/main/website/docs/tutorial_torchhub_detection_inference.md)
- [Official SlowFast checkpoint factory](https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/slowfast.py)
- [AVA action labels](https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt)
- [AIRTLab dataset](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)

Completed source traces are saved individually. Reuse requires an identical code/input fingerprint and checkpoint hash. If code, source model, or protocol changes, use a new output directory rather than mixing runs. `approved_for_demo` and `real_phone_tested` remain false until separate acceptance checks.
