# SEM fiber analysis

## Run everything

From `C:\Users\zvallarino\code\clement` in PowerShell:

```powershell
.\run.ps1
# Explicit folder and output location:
.\run.ps1 ".\pictures\TIFF" -o ".\output\full_review"
# One image:
.\run.ps1 ".\pictures\TIFF\26-30_NF3-12%PVP_0004.tif"
```

The full runner performs metadata/calibration checks, banner removal, Depth
Anything V2 **Large**, native-pixel-checked axes and diameters, independent non-AI diameter measurement, visible-path counting,
and density/coverage analysis. It uses the current `run_geometry.py` and
`classical_diameter.py`, and `run_density.py`, not the old combined geometry algorithm. Folders are scanned
non-recursively for `.tif` and `.tiff`; TXT files are not processed as images.
Multiple inputs are supported and duplicate image paths are removed.

`run.ps1` launches `run_pipeline.py`, whose CLI delegates to `run_all.py`.
With a working Python environment, the equivalent command is:

```powershell
python run_pipeline.py pictures/TIFF -o output/full_review
```

The old `--depth-model large` command still works. Large is now the fixed default,
with model revision `7581137eff8d4e94f6e796d3baea0e9fa79b22d2`; Small/Base are
not substituted. The model is cached across images. The count sensitivity sweep
adds work per image, so the full pipeline takes longer than geometry alone.

## Results

Each run creates a new `output/full_review/batch_<timestamp>` folder:

- `batch_summary.csv`: one row per input, geometry measurement totals/review,
  non-AI measurement totals and method agreement, central path count, sensitivity range, density, coverage, errors, and output links.
- `manifest.json`: inputs, options, model revision, environment/dependency versions,
  and Python/launcher source hashes.
- `quick_review`: `__geometry.png`, `__non_ai.png`, `__segmentation.png`, and `__count.png` review images.
- `001__<image>/geometry/<timestamp>/`: current geometry outputs.
- `001__<image>/non_ai/<timestamp>/`: independent diameters, segmentation, and method comparison.
- `001__<image>/density/<timestamp>/`: count/density outputs.
- `001__<image>/image_summary.json`: combined per-image result.

Geometry files include `geometry_preview.png`, `source_geometry_preview.png`,
`large_depth.npy`, `preview_notes.json`, `diameters.csv`, `diameters_audit.csv`,
`diameters_ok.csv`, `measurement_summary.csv`, `review_reasons.csv`, and
`REVIEW_README.txt`. Use `diameters_ok.csv` for software-accepted widths;
`diameters.csv` also retains measurable rejected candidates for review.
Widths use micrometers when calibration is available, otherwise explicit source
pixel units. The audit includes coordinates, provisional track IDs, and reasons.

Density files include `density_summary.csv`, `density_report.json`,
`count_sensitivity.csv`, `count_candidates.csv`, `count_overlay.png`,
`count_low_overlay.png`, `count_high_overlay.png`, and `READ_ME.txt`.

The batch summary is saved between geometry, non-AI measurement, and density for each picture.
An exception in one stage does not prevent the other stage or later images from
running. `status=partial` means some stages completed; `failed` means none completed. Copy errors are reported separately. Nonzero exit status indicates
an execution/output failure. Low measurement totals are review flags, not crashes.

## Interpret review, count, and density

`completed` means processing finished, not that an analyst verified accuracy.
`measurements_total` includes all numerical widths exported; `measurements_ok`
passed the software checks and `measurements_check` require review. Red lines
are a spaced subset of final OK widths after native-pixel and neighbor review,
so their count can be lower than the number of OK measurements.

`review_status` is `no_ok_measurements`, `below_target`, or `target_reached`.
The default target is 100 OK cross-sections per image; reaching it is not a
validation criterion. Multiple sections on one track are not independent fibers.
`review_reasons.csv` explains rejected widths and separately lists geometry
candidate rejections. A width can have several reasons; do not add these counts
as though they were unique measurements.

The count module estimates visible stitched paths. Geometry F IDs and count P IDs
are independent and must not be joined as though they identify the same fibers.
The central count uses the default setting. The low/high range is the minimum
and maximum over a 3-by-3 sweep of segmentation and continuation settings, **not
a confidence interval**. Path IDs can change between settings. Inspect the
labeled overlays for split, merged, duplicate, or missed paths. Retained border
paths count; paths shorter than the configured visible-length threshold do not.
Blank researcher fields in `count_candidates.csv` support manual review; edits
are not automatically applied to the count.

Physical density is visible paths divided by the cropped field area in square
micrometers. `automatic_paths_per_megapixel` uses the cropped **source** pixel
area, so its value depends on acquisition resolution. Coverage is projected
foreground area percentage; neither measurement is mass density or 3D porosity.
Missing calibration leaves physical density empty. The count module always
reports `researcher_review_required`, even if the diameter target was reached.

The counter remains experimental: an earlier manually counted 20–21-fiber image
produced a 20–57 automatic path range with incorrect crossing assignments. Batch
integration does not establish count accuracy. The Large model proposes relative
depth; it does not establish validated physical heights or definitive crossing
order. Ambiguous/hidden boundaries are omitted. No 99% accuracy claim is made.

## Inputs and calibration

Original TIFFs belong in `pictures/TIFF`. Matching JEOL TXT metadata may sit beside
the image or in `pictures/check`; duplicate matches are rejected. Text metadata
check image dimensions, calibration information and banner crop. Conflicting
explicit crop settings fail rather than silently overriding metadata. When a
sidecar is absent, the modules use their recorded TIFF magnification/reference
calibration if available. Nominal calibration should be checked independently.
Site discovery and non-AI segmentation use a maximum image dimension of 1600 pixels.
Geometry rechecks proposed edges by sampling the original TIFF at native resolution,
with the smoothing footprint and review tolerances preserved in analysis coordinates. Fine fibers in
wide fields can become too narrow to measure reliably; more red lines or a higher
sampling target does not fix this.

## Settings and individual stages

```powershell
# Full run: target is a reporting threshold, not a quota forced into geometry
.\run.ps1 pictures\TIFF --target-measurements 100 --threads 4
# Geometry only
.\run_geometry.ps1 pictures\TIFF
# Count/density only, using a working Python environment
python run_density.py "pictures\TIFF\26-30_NF3-12%PVP_0004.tif"
```

Full-run geometry controls: `--nearest` (15), `--sample-step` (4 analysis pixels),
`--crossbar-spacing` (6 analysis pixels), `--target-measurements` (100).
Count controls: `--pair-score` (0.55), `--pair-delta` (0.15),
`--threshold-delta` (0.03), `--min-visible-fraction` (0.05 of image diagonal).
`--crop-bottom` applies to both stages; `--threads` controls Torch CPU threads.
The count module itself additionally supports an independent
`--reference-count LOW HIGH` and `--nm-per-px` for a single image. Those overrides
are not accepted by the full runner, avoiding accidental application of a single
image's count or calibration to an entire folder.

Geometry requires stable source boundaries, checks center shifts, missing edges,
local width changes, merged seams and track outliers. Export checks flag widths
that differ from neighboring measurements by more than 20%, or lack sufficient
neighbors. These are heuristic checks that can reject valid sections or accept
incorrect ones. Consult `preview_notes.json` and the source overlay.

## Independent non-AI measurement and accuracy checks

Every full run now includes `classical_diameter.py`. It selects sites independently
of the depth model using a binary source-image mask, skeleton centerlines, local
orientation, and distance maps. It measures interpolated mask boundaries along a
perpendicular section, rather than rounding twice the integer distance-map radius.
The threshold is midway between background and bright-class median intensities;
this avoids a background-edge threshold bias found in the synthetic tests. It is
an experimental implementation inspired by classical methods, **not DiameterJ**.

A candidate needs stable boundaries under threshold changes of +/-0.03 and stable
neighboring sections. Widths below 10 analysis pixels, crossings, endpoints,
ambiguous directions, off-center sections, weak contrast, and internal seams are
excluded. Crossings are excluded within 1.25 local widths. These are documented
heuristics, not calibrated uncertainty limits. The seam rejection helper is shared
with geometry; the two methods are not fully independent sources of truth.

Files in `non_ai`:

- `non_ai_measurements.csv`: sampled candidates, widths where measurable, status and reasons.
- `non_ai_ok.csv`: only accepted non-AI widths.
- `non_ai_overlay.png`: accepted red widths and local blue axis marks.
- `segmentation_mask.png` and `segmentation_overlay.png`: the actual mask and its boundaries.
- `non_ai_report.json`: settings, coordinates, calibration, rejection reasons and statistics.
- `non_ai_histogram.csv`: diameter distribution, when accepted measurements exist.
- `method_comparison.csv`: checks at exactly the geometry method's final OK locations.
- `cross_checked_measurements.csv`: only same-location agreements. Widths here are explicitly source pixels.

Comparison is `agree`, `disagree`, or `not_comparable`. Agreement requires no more
than 10% relative width difference (using the mean width as denominator) AND
agreement of both edge positions within 10% of the geometry width, with a one
analysis-pixel floor. An unresolved section is never counted as agreement. The
batch columns `method_agree`, `method_disagree`, and `method_not_comparable` count
these categories. When geometry fails, the non-AI method still runs, but there is
no geometry comparison. Do not pool the methods' rows: sites can overlap and
sampling populations differ. Neither method's OK label is analyst confirmation.

Geometry v4 rechecks previously accepted sections on original TIFF pixels. The
sampling coordinate transform preserves pixel-center alignment and anisotropic
resize factors. Native verification uses the same physical smoothing scale and
review tolerances as the preview, avoiding spurious changes caused by pixel scale.
It interpolates gradient peaks, records original preview endpoints, and rejects
unresolved or substantially changed widths. It never promotes a previously
rejected candidate. Final red lines now correspond to final OK CSV measurements.

The non-AI method can run without Torch or a depth model:

```powershell
python classical_diameter.py "pictures/TIFF/image.tif"
# Optional same-location comparison with a completed geometry output folder:
python classical_diameter.py "pictures/TIFF/image.tif" --geometry-output "output/geometry/<image_result>"
```

Known-width, rotated, adjacent, crossing, bright-rim, low-contrast, blank, and
underresolved synthetic cases are tested, along with native coordinate scaling
and method-comparison states. Real-image smoke tests demonstrate execution and
reviewability, not a measured clinical/QC accuracy rate. Very dense wide-field
images may still produce no accepted measurements. Increasing a sampling quota
cannot recover boundaries absent at the analysis resolution.

## Python setup, portability, and tests

The PowerShell launchers first try `.venv\Scripts\python.exe`. If it is broken,
they try the locally installed Codex Python 3.12 runtime with existing project
packages. That fallback may not exist on another computer. For a portable setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-depth.txt
.\run.ps1
.\.venv\Scripts\python.exe -m unittest discover -v
```

The first Large-model run needs the model weights available locally or a download.
Pictures, outputs, model caches, and virtual environments are ignored by Git;
copy original TIFF/TXT data separately when moving computers. Do not expect a
Git clone to include the images or model weights.

`run_pipeline.Config` and `analyze_one` remain importable for historical tests and
comparisons. They are not the current CLI path. `REVIEW.md` records the historical
review and proposed validation work; manual comparisons still require a separate
reference protocol.
