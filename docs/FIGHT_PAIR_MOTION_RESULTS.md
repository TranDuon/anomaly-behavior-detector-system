# Pair/motion v1: do not promote

Run: 27–28 October 2026 (Asia/Bangkok), training revision
`5fb5fb13c992676dc898d96971b7b1bb02c23e97`, NVIDIA Tesla T4.
[Saved public artifacts](https://drive.google.com/drive/folders/1UtR82niSXTiS875nkbkZEe6DVTyZSQQi).
The allocated runtime was released automatically after 713 seconds from the first
budget cell, including authorization/setup/staging. The training runner itself took
194.60 seconds, including motion preprocessing and export. No additional GPU run
was started after the no-progress stop. The authorized maximum was 25 minutes.

## Frozen public protocol

Both variants use the matching large-v1 frozen RGB feature cache, not the fine-tuned
backbone of the currently installed RGB-aux model. They therefore form a matched
ablation, not a direct claim to improve on the installed model. Train/development/
calibration/test counts are 1526/216/228/202. Seed 751; 512 sampled exposures per
epoch with replacement. Both stopped after four completed epochs because neither
met the predeclared protected development improvement rule.

| Metric | RGB control | Pair + motion |
| --- | ---: | ---: |
| Development worst precision/recall | 0.7500 | 0.7500 |
| Development macro AP | 0.935640 | 0.935626 |
| Independently calibrated threshold | 0.404590 | 0.552028 |
| AIRTLab diagnostic accuracy (20 windows) | 75.00% | 75.00% |
| AIRTLab precision / recall | 55.56% / 83.33% | 55.56% / 83.33% |
| AIRTLab TP / TN / FP / FN | 5 / 10 / 4 / 1 | 5 / 10 / 4 / 1 |
| RLVS diagnostic accuracy (182 windows) | 95.60% | 94.51% |
| RLVS precision / recall | 95.70% / 95.70% | 95.60% / 93.55% |
| RLVS TP / TN / FP / FN | 89 / 85 / 4 / 4 | 87 / 85 / 4 / 6 |

Selection was frozen using development only before diagnostic test scoring; it selected
RGB control. Pair/motion did not meet protected progress over that control. The test
split has been inspected in previous experiments; paired AIRTLab camera views are
not independent demonstrations. None of these numbers is physical camera accuracy.

## Checks and limits

Six deterministic cached examples passed exported CPU versus original GPU score
agreement at absolute tolerance 1e-4, including unchanged frozen-threshold decisions.
The maximum observed score difference was 1.20e-7.
That checks serialization/preprocessing, not action accuracy or real-time throughput.
Runtime supports the same three-input preprocessing for uploaded files and live input.
Private phone recordings remain local and are not part of the training/calibration
data or the public artifact package.

An additional whole-video CPU API diagnostic used five deterministic public sources:
three AIRTLab hard negatives and one positive each from AIRTLab/RLVS. Pair/motion
produced TP=2, TN=2, FP=1, FN=0 through both uploaded-file and paced replay paths.
One greeting/hug/jump source still triggered a false alarm. Matching observed windows
had identical scores and equal event counts, but replay freshness failed: maximum
observed prediction age reached 3.102 seconds. The installed RGB-aux model also failed
freshness in the same diagnostic (maximum 2.899 seconds). Equal event counts do not
prove equal intervals or action correctness. No camera hardware was opened by this
API test, and five clips cannot establish live camera accuracy.

The installed RGB-aux pilot is deliberately retained. Adding geometric descriptors
and a small motion head to frozen legacy features did not establish a recognition
improvement. This short negative result does not prove optical flow or pose are
universally ineffective, nor does it identify dataset noise as the sole cause.

Before another long run, the next controlled comparison should start from the current
stronger RGB-aux backbone (with newly fingerprinted features and its original classifier
preserved), and validate action-window labels versus hard negatives rather than just
whole-clip any-alarm labels. Keep a separate untouched capture-domain evaluation set;
do not train and report acceptance on the same two already-inspected recordings.
Measure missed events, false alarms per normal minute, and CPU observation freshness
separately. Lowering the threshold to make a particular example red is not acceptance.
