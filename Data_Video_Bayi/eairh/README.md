# Eairh collection

`source_clips` contains **14 videos**: 13 fuller DVD shots (75.6 seconds total)
and one four-second uploader-labeled short-video candidate. Existing DVD IDs retain
their meaning; the manifest's `shot` field gives chronological order. These are
edited shots, not necessarily individual vocalizations or unique babies.

## DVD exports

`segments.json` is the current cut list. `source_clips/provenance.json` records
the actual exports, hashes, durations, audio RMS and source identity. Boundaries
omit black transitions. The two shots with captions are cropped to remove them.
`ea_0006` is retained for completeness but marked `training_eligible: false`
because of low light and a small subject.

The earlier eight DVD excerpts were replaced after the complete new set passed
audio/video timing, non-silent audio and black-transition checks. Previous exports
and their provenance are archived under `artifacts/dvd_previous_*`. The short-video
clip and adjacent provenance JSON were not changed.

The source is `VTS_01_1.VOB` (without `(1)`). Its Lesson Two explicitly identifies
the fourth word as eairh and describes lower wind pain. Labels derive from that
instructional section, not independent confirmation of each baby's condition.
See [the full timestamp review](dvd_full_review.md) for original shot boundaries.

## Reproduce

Raw DVD timestamps reset, so normalize sequentially before cutting:

```powershell
ffmpeg -v error -i "VTS_01_1.VOB" -map 0:v:0 -map 0:a:0 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart -n artifacts/source_normalized.mp4
.venv/Scripts/python extract_eairh.py --replace
```

Skip normalization if the normalized copy already exists. Without `--replace`, the
extractor refuses existing outputs. Replacement validates all new clips in staging
and archives prior DVD files before replacing them; it does not touch the short.

## Dataset status

All DVD shots share `dunstan_lesson_two_vts_01_1`. The accepted short candidate
`ea_short_AhNLgj4JHU4.mp4` has separate provenance and retains its uploader-label
status. `split_manifest.json` now assigns 9 clips to training, 2 to validation,
and 2 to testing; the dark clip is excluded. Copies are in `data_train_video`,
`data_val_video`, and `data_test_video`. The source archives' unassigned fields
describe their extraction state; the split manifest is authoritative for dataset use.
No retraining has occurred.

This is a provisional within-DVD split: visibly recurring shots are kept together,
but DVD source independence and infant independence are NOT established. The
uploader-labeled short is training-only. Validation/test each contain just two
eairh samples. Reproduce with `.venv/Scripts/python split_eairh.py`.

`cries.py prepare` honors explicit `data_val_video` folders, falling back to the
existing seeded validation split for other classes. It uses only the first four seconds of each input, so these
longer preserved shots will need utterance/window preparation to use all their audio.
That future preparation must respect source groups and quality exclusions.

Generate new five-class features separately from the old four-class artifacts:

```powershell
.venv/Scripts/python cries.py prepare --output artifacts/five_class
.venv/Scripts/python cries.py train --features artifacts/five_class/features.pt --output artifacts/five_class/fusion
```

Expected combined counts: 97 train, 26 validation, 30 test. These commands have not
been run as part of splitting; existing cached features and checkpoints are unchanged.
