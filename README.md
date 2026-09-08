# SEM fiber analysis — reviewed revision

This revision integrates calibration, banner removal, segmentation, path tracing,
local crossing cues, axes, repeated diameter measurements, and density reporting.
Results are provisional and require image review. No 99% accuracy claim has been
established. `accepted` means a measurement passed software filters, not that an
analyst has confirmed it. `review_required` is the current image-level status.

## Run from the Clement folder

```powershell
.\run.ps1
# Or specify inputs and output folder:
.\run.ps1 pictures\TIFF -o output\review
```

The existing `.venv` points to a missing Microsoft Store Python installation.
The launcher first checks it, then uses the installed Codex Python 3.12 runtime
with the existing 3.12 packages. It does not repair or modify the old environment.
For a portable installation, install Python 3.12 and create a fresh environment:

```powershell
py -3.12 -m venv .venv-review
.\.venv-review\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-review\Scripts\python.exe run_pipeline.py pictures\TIFF -o output\review
.\.venv-review\Scripts\python.exe -m unittest discover -v
```
Dependency versions record the environment exercised during this review. A new
installation and other platforms have not been tested. Torch/Transformers are
optional and listed separately in `requirements-depth.txt`.

## Parameters and measurement definitions

```powershell
# Explicit crop and independently established calibration:
python run_pipeline.py image.tif --crop-bottom 256 --nm-per-px 2.48046875
# Recorded segmentation override (grayscale intensity from 0 to 1):
python run_pipeline.py image.tif --threshold 0.15
# More tracing resolution; diameters already use native-resolution pixels:
python run_pipeline.py image.tif --max-dim 2400 --max-samples 500
```

- **Estimated visible paths/image:** traced fiber paths after stitching and
  length filtering. Edge fragments are included. Splits, merged fibers, hidden
  segments, and touching fibers make this an estimate rather than a true count.
- **Paths/µm²:** estimated paths divided by the cropped field's physical area.
  Compare images acquired at the same field size/magnification: even normalized
  counts depend on field boundaries and the visibility of underlying fibers.
- **Paths/megapixel:** the same estimate divided by original cropped pixels and
  multiplied by one million. This is not invariant to magnification or resampling.
- **Coverage (%):** foreground fraction in the analysis mask, a separate 2D
  measure. It is not volume fraction, mass density, or 3D porosity.
- **Diameter:** native-pixel, perpendicular edge-to-edge distance at systematically
  spaced visible locations. Both sides must leave the mask into background with
  adequate contrast. Unmeasurable locations are retained with rejection reasons.
  The summary is the median of accepted cross-sections, not a fiber-weighted
  population mean. Visible isolated fibers can be sampled preferentially.
- **Axis:** global principal direction per reconstructed path, and a local
  direction per diameter location, in degrees modulo 180 from the image x-axis.
  Image y increases downward. Curved paths do not have one constant local axis.
- **Order:** local intensity/occlusion cues, with weak cases left unresolved.
  Scores are heuristic cue separation, not probabilities. There is no assumed
  global stack order: flexible fibers can weave over and under each other.

Calibration uses the existing JEOL 127 mm reference and original image width.
Confirm it independently against an actual scale bar or instrument calibration.
A missing calibration leaves physical fields empty; no silent 1 nm/pixel default.
The original 5120 × 4096 inputs in this review crop to 5120 × 3840. Calibration
always refers to original pixels. Crop/resize transforms are recorded explicitly.

## Output in a new timestamped run directory

`summary.csv` contains one row per input, including errors. `manifest.json`
records configuration, Python/package versions, and source hashes. Each image has:

- `report.json`: source SHA-256, calibration, ROI, exact resize factors, thresholds,
  mask sensitivity, metric definitions, and review flags.
- `diameters.csv`: accepted and rejected sites, endpoint coordinates in original
  pixels, units, local axes, and contrast/coherence values.
- `fibers.csv`, `centerlines.csv`, `crossings.csv`: reconstructed paths and cues.
- `review_overlay.png`: path labels and red accepted diameter segments.
- `cropped_analysis.png`, `fiber_mask.png`, `native_fiber_mask.png`: image and masks.
  White in a binary mask means fiber. Native mask is written when measurements run.
- `manual_reference_template.csv`: sample locations with blank analyst/session/
  reference-diameter fields, deliberately omitting automated values.

Do not process colored overlays as inputs. Supply original grayscale images.
Automatic segmentation separates three intensity classes (background, dim fiber
bodies, bright rims) and retains the upper two. This is an unvalidated heuristic,
not a universal solution. Binary Otsu and Triangle coverage are retained as
sensitivity checks, with disagreement flags. Review the saved masks.
The minimum path length and sampling spacing are analysis-pixel parameters;
freeze resolution/settings in any comparison study. Under-resolved tracing is flagged.

## Optional depth visualization

```powershell
python run_pipeline.py image.tif --depth-model small --depth-revision 5426e4f0f36572d16453bbda7a8389317b1bef99
```

The model runs on the unpainted cropped image and saves floating-point relative
depth plus model revision metadata. It is not used as measurement truth or to
force a crossing order. The cached Small model was executed successfully during
review; this does not validate its SEM predictions. A missing model may require
a first download. Model failure is recorded without discarding conventional
measurements. Standalone `top_fiber_depth.py` remains available for highlighting.

## Manual / DiameterJ comparison

Use original TIFFs in Fiji/ImageJ with independently checked calibration. Give
analysts the source image and location template, without automated overlays or
diameters. Have them record perpendicular edge-to-edge diameters in nm, analyst
ID, and session. Combine completed rows into a reference CSV and run:

```powershell
python compare_reference.py output/review/run_TIMESTAMP references.csv -o comparison --tolerance-pct 10
```

The 10% tolerance is an example, not a validated acceptance criterion. The tool
reports paired errors and analyst-specific descriptive bias/agreement; it rejects
duplicate readings and unmatched sites. Use only the matching run: sample IDs
can change with settings. Repeat readings must have distinct session IDs.

Paired accepted sites test diameter placement but miss selection/detection bias.
Also use independently selected fields/sites, consensus path/crossing annotations,
and reference wires/digital phantoms. Freeze field selection, crop, calibration,
segmentation settings, endpoint conventions, versions, and rejection rules before
a held-out study. Include different concentrations, magnifications, days, analysts,
and microscopes/labs. Report rejection rate alongside measurement error; abstaining
on difficult cases cannot be counted as 99% successful automated analysis.

DiameterJ is a relevant independent semi-automated comparator. Its published
validation does not transfer automatically to these specimens and settings.
Use matched fields and units, retain its segmentation choices, and distinguish
its diameter-distribution statistics from this tool's sampled median.

Sources: [DiameterJ documentation](https://imagej.net/plugins/diameterj),
[NIST DiameterJ](https://www.nist.gov/mml/bbd/biomaterials/diameterj),
[Depth Anything V2 model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
#   L e t s  
 