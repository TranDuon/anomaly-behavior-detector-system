# Independent action review backend

`mmaction_separate_actions` is a side-by-side experimental backend, not a claim of solved classroom recognition.

| Label | Source |
| --- | --- |
| strike | Original AVA `fight/hit (a person)` score per person |
| fall | AVA `lie/sleep` score per person; demo includes already lying and moving into lying |
| grapple | Existing separate grapple pilot head; not AVA grab renamed as grappling |
| fast_running | Existing running checkpoint and AVA/motion applicability checks, unchanged |

Each label has its own development threshold, confirmation gate, score, evidence regions and event journal entry. Grapple and fall may overlap without being relabeled as striking. The old combined checkpoints remain available and are not overwritten.

On 2026-11-11 the user explicitly redefined `fall`: moving into lying or already lying, including intentional lying and lying on a bed. It is not an accident claim. Sitting, bending and kneeling alone do not satisfy the new definition. The separate backend now uses AVA class 8 (`lie/sleep`), not class 5 (`fall down`), with the existing AVA label-display cutoff of .4 and the existing .4-second confirmation gate. This cutoff is explicitly uncalibrated for the new task. No retraining or weight changes are made. A single-frame spike cannot trigger an alert; confirmation still needs a full RGB clip and repeated support. Missing person detection remains insufficient evidence, not proof that nobody is lying.

The action engine adds only the raw class-8 score to its output. The separate-backend loader accepts the old calibration fingerprint only when removing exactly this additive output line reproduces it; other feature-engine changes still block startup. The strike/grapple/running calibration is retained, but the old fall metrics and threshold are not presented as validation for the new lying policy. The GUI defaults to the separate backend and explains this definition; historical backends and checkpoints are unchanged.

Policy v2 adds a posture-only confirmation hysteresis after a short bed clip exposed score wobble: .705 at 2.141 s followed by .336 at 2.676 s previously canceled confirmation. A strong .4 observation is still mandatory. A following observation may continue confirmation above .28 only when the current and prior supported person boxes are horizontal (width/height >= 1.5), overlap with IoU >= .35, and AVA still supports lying for that person. Box shape alone cannot initiate an alert, a different person cannot provide this weak continuation, and ordinary action gates keep their old behavior. Raw scores remain unchanged; `fall_confirmation` logs the explicit continuation decision. This is a provisional posture rule, not a calibrated accuracy claim. Source changes, missing people and loss of supporting geometry reset it.

Policy v3 also accepts the reverse order: a weak observation >= .28 followed by a strong observation >= .4 may confirm when the same horizontal-box overlap checks pass and the source-time gap is within the gate freshness limit. The earlier observed timestamp supplies the confirmation start; no synthetic frame, end-of-file promotion, or additional model score is generated. This addresses a second short bed clip whose final two scores were .327 and .459. Two weak observations alone still cannot trigger an alert. `previous_support_used` makes this behavior visible in the trace.

## Current-frame posture override

The separate backend's final `fall` output now comes from `fight/lying_posture.py`, not AVA class 8. Existing local YOLOv8n-pose weights locate COCO joints on each sampled current frame (640 input, CPU), including while the RGB buffer is still collecting. The hand-written rule requires a visible shoulder/hip/knee/ankle chain, horizontal torso and body, an extended leg aligned with the torso, and two spatially consistent observations .15–.8 seconds apart. Missing joints do not count as lying. Sitting/bending are explicitly tested as negatives. Bounding-box shape alone never initiates an alert. The binary posture value is rule evidence, not a probability or measured accuracy. Perspective, occlusion and low-confidence pose can still fail.

RGB strike/grapple/running inference is unchanged. The legacy AVA fall score/gate remains internal for historical code compatibility but is overridden before recording/display in this backend. Pose runs at the existing 3.75 FPS input sample cadence, not every display frame; there is no claim of zero latency. No training or remote upload is involved. Warmup/model loading still takes time before capture starts. The raw pose probe first confirmed the already-lying floor clip at .50 s and the bed transition clip at 2.65 s; these are source timestamps on two test clips, not universal guarantees.

Pose display uses a single current-frame pose person inventory rather than appending unmatched RGB boxes. RGB labels are transferred only to matched current pose boxes; unmatched labels/regions are hidden. If no pose person is available, only a fresh same-frame RGB detector inventory may be displayed as fallback. This may omit a pose-missed person, but does not invent a second person from a stale detector position. Fall regions retain the exact canonical pose coordinates so confirmed alerts render red and optical-flow display tracking can preserve their attribution. The backend still cannot guarantee identity matching under crowd occlusion.

When full-frame pose misses lying evidence, up to two horizontal detector ROIs are expanded and re-run through the pose estimator. Only a successfully reconstructed joint chain can recover a candidate: the detector box alone never establishes lying. Recovery ROIs expire after .8 seconds and are not themselves drawn as extra people. Empty person inventories after RGB warmup are reported as insufficient, not normal. A replay of the bed clip confirmed at 2.68 seconds and retained a pose at the final frame, but intermittent pose loss still split the event into multiple intervals; this remains a stability limitation, not an accuracy acceptance claim.

The perspective review adds a second explicit pose rule for a foreshortened reclined torso: a confident shoulder–hip–knee–ankle chain, a projected leg >= 2.2 torso lengths, aligned torso/leg directions and diagonal body support. It does not treat arbitrary sitting with straight legs as lying. This rule is camera-dependent and provisional, not perspective-invariant geometry. Pose person records now require visible shoulder and hip support plus six sufficiently confident joints, filtering the low-confidence right-border phantom in the reviewed room video. The two-sample confirmation uses overlap of pose-validated candidates instead of reimposing a horizontal-box ratio that incorrectly rejected diagonal lying. Weak detector-only fallback is suppressed after pose has provided the current-frame inventory; rejected old boxes must not reappear from that same pose-enriched output. Some real partially occluded people may also be hidden by this conservative reliability filter.

`mmaction2_demo/evaluate_separate_actions.py` scores the existing reviewed public corpus and compares AVA strike/fall against the failed three-head run. It selects thresholds on development only, preferring precision >= .8 then F1; test is reporting only. This is a repeated, small AI-assisted review set, not new independent ground truth. Its scores cannot establish camera accuracy. Temporary public videos are checksum-verified and removed after extraction; private iPhone clips are not uploaded or used for training.

`separate_actions_calibration.json` pins the AVA source, engine, preprocessing and grapple/running checkpoint hashes. A missing or incompatible calibration blocks startup. Lowering a threshold for one failed demonstration video is not part of the procedure.

Pending work: independent human review of temporal action intervals and person/pair attribution, a dedicated fall model evaluated against AVA fall, larger grapple data, and event/latency acceptance on real classroom camera streams. The researched GajuuzZ TSSTG official Google Drive download returned 404 on 2026-11-11; no replacement weight is fabricated or silently substituted.

## Pose experiment reviewed on 2026-11-11

The completed public trial is `pose_specialists_v1/run_20261110_181122` on Drive; its run tag uses the Colab clock, not the user's local date. Both checkpoints were downloaded to `mmaction2_demo/results/pose_specialists_20261111/verified`, checked against the run's source/SPEC fingerprints and checkpoint checksums, and loaded strictly with `weights_only=True`. Neither was installed into the web backend.

The source is the official MMAction2 v1.2.0 ST-GCN NTU60 2D checkpoint, SHA-256 `484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea`. It is general action pretraining with a falling class, not a validated classroom fall alarm. The adaptation uses dense 15 FPS YOLO pose, 32-frame tracks, a temporal pair-relation branch for grapple and an independent fall output. Its pose estimator differs from the HRNet preprocessing of the source dataset.

| Branch / development-selected configuration | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: |
| Grapple / last two ST-GCN stages adapted | .667 | .286 | .400 |
| Fall / frozen ST-GCN | .333 | .143 | .200 |
| Existing RGB grapple baseline | .500 | .071 | .125 |
| Existing AVA fall baseline | .500 | .143 | .222 |

These are repeated reviewed windows, not camera accuracy or independent acceptance. Grapple catches more positives but remains weak; fall does not improve. Both pose experiments peak on development at epoch 1 and deteriorate afterward, so extending the same training is not justified. Only three positive grapple recording groups and four positive fall groups are in training. Eleven of 141 windows have no sufficiently complete pose track, including several fall transitions; this is an upstream tracking/visibility failure, not something a lower action threshold can recover.

`review_pose_specialists.py` makes **per-label** development selections. The original trial's single macro-AP selection must not replace both branches: the configuration preferred for grapple degrades fall. Test metrics are reporting only. Further work needs more independently reviewed recording groups, temporal/person attribution, improved pose continuity through falls and occlusion, and a source-compatible dedicated fall baseline. Do not silently turn missing pose into normal activity or promote these checkpoints.

## Startup and warm reuse

### Depth-aligned lying geometry (2026-11-12)

Fall policy v5 adds a separate high-confidence COCO-joint rule for front/back depth-aligned lying. It requires a torso compressed relative to shoulder width, an extended leg aligned with the torso, and visible bilateral shoulders/hips. This complements the existing horizontal and oblique rules; it does not classify lying from a wide detector box alone. Two observations of the same person remain required. Strike, grapple, running weights and decision thresholds are unchanged.

The local frontal-bed regression previously produced no lying evidence; v5 confirms lying from 2.13 seconds through 7.47 seconds and does not confirm the later seated/standing interval. This is a development regression, not independent accuracy validation. Occluded joints, sharply bent legs, camera roll and extreme perspective remain limitations; 2D pose cannot reliably resolve every lying-versus-sitting ambiguity. No training or new checkpoint is involved.

### Strike regression repair (2026-11-11)

The separate backend had used a strike threshold of .24135, unlike the earlier original-public threshold .02952. It now explicitly reuses the legacy original-public strike calibration, validating the original checkpoint SHA and clip SPEC and retaining the calibration provenance. This is not a fresh validation of the changed person detector, nor a camera accuracy claim. Original weights and running thresholds remain unchanged.

Full-body pose visibility must not veto the RGB inventory: close-up people may have no visible hips. The host retains a separate RGB result snapshot, merges pose lying evidence only for display, expires old RGB geometry, and suppresses active labels without localized regions. Pose refreshes no longer feed their own modified boxes back into the RGB snapshot. The weak grapple pilot keeps diagnostic scores but no longer emits an alarm in this backend; its previous public recall .071 and observed strike/headlock confusion do not justify presenting it as a verified headlock classifier. Historical backends remain available unchanged.

During collection the GPU worker now runs a detector-only request on the first frame and periodically until all nine action samples are available. It does not feed repeated frames into the action model, advance action confirmation gates, or invent action scores. Status reports sample progress; approximately 2.13 seconds of source video is still necessary for the original action clip. A local public-frame smoke check detected three people in .149 seconds **after** a 16.47-second cold model load. This is one measurement, not a universal latency guarantee.

After a completed MMAction video, the worker stays warm for the next test with the same backend. A new source resets samples, scores, gates and session events without reloading the weights. Explicit Stop, backend changes, errors and application shutdown close the worker. Warm reuse occupies local GPU memory while idle; it does not keep a camera open or retain the previous action state. Camera/upload acceptance and end-to-end latency must still be checked separately.
