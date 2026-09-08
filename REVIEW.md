# Code review — 8 September 2026

All five original Python files, the README, requirements, and all 13 starting
TIFFs were inspected. Source originals were staged without changing the inputs.
The project had no test suite or integrated density runner. An existing output
overlay is not ground truth; the reported 85% success rate has no supplied
definition or annotated benchmark. No improvement to 99% can be demonstrated yet.

| Original finding | Consequence | Revision |
|---|---|---|
| `fiber_diameter.walk_to_edge` searched the strongest drop anywhere along a ray | Diameter could include a gap and neighboring fiber | Anchor to first mask exit, require sustained background, refine nearby derivative |
| Three incompatible banner rules, including first dark row and fixed 260-pixel crop | Real specimen rows removed or banner left behind | Shared conservative bottom-band detection, validated crop override, recorded ROI |
| Pillow `convert(L)` on high-bit TIFFs | High intensities clipped rather than normalized | Shared finite-range 16-bit normalization |
| Default calibration 1 nm/pixel | Plausible but wrong physical diameters | Metadata calibration or explicit override; missing physical values stay empty |
| Magnification accepted zero/nonfinite values | Invalid scale/crashes | Positive finite validation; nominal bar label no longer called verification |
| Hungarian assignment converted greedily into undirected pairs | Cycles/incorrect pair selection | Actual undirected maximum-weight matching |
| First branch not oriented toward its linked end | Backtracking and bridges across the wrong span | Orient first branch using its link |
| Unbounded transitive crossing merging | A dense field collapsed into a giant crossing region | Spatial/member bounds; dense-case profiling regression |
| Full-frame scan for every skeleton branch and repeated intensity scans for every crossing | Avoidable time growth | Component bounding boxes, center-of-mass labels, per-fiber intensity cache |
| Per-fiber full-image distance stack in renderer | Memory grew as fibers × image pixels; empty stack crashed | Shared nearest-centerline transform and empty-image handling |
| Recoloring all intersections of a fiber pair using each local verdict | One crossing's order applied to other crossings | Limit recoloring to the local crossing region |
| Every crossing assigned top/bottom, including indistinguishable profiles | False certainty | Unresolved results on weak/parallel evidence; scores labeled heuristic |
| One best-looking diameter selected | No distribution, selection audit, or per-image repeatability check | Systematic spaced sites, per-site rejection log, original-coordinate endpoints |
| Otsu sometimes segmented bright rims and excluded darker fiber bodies | Artificial paths and rim-width measurements | Three-class background split, alternative thresholds retained, disagreement flags |
| Depth model fed red overlays; raw-image fallback treated entire frame as fiber | Painting/background could drive predictions | Original grayscale input, separate mask, finite shape checks and model caching |
| Depth tensor might have singleton batch dimension | PIL resize failure | Explicit tensor-to-CPU conversion and squeeze/validation |
| Depth save path could overwrite output when extension was not `.png` | Lost visualization | Suffix-safe paths and numeric depth output |
| Missing `tifffile`/optional model dependencies and broken `.venv` interpreter | Setup did not reproduce the scripts | Tested versions, optional depth list, non-destructive local launcher fallback |
| No unified runner or density calculation | Inconsistent chained inputs and no auditable final metric | `run_pipeline.py`, per-image batch status, hashes, masks, CSV/JSON and defined units |

## Evidence and limits

Synthetic regressions test known-width horizontal/rotated fibers, adjacent-fiber
edge jumping, dark interiors, blank images, crop rejection, TIFF dynamic range,
scale validation, unresolved crossings, branch orientation, bounded zone merging,
coordinate scaling, count/area units, and reference comparison input checks.
The real-image run is a processing and visual review exercise, not an accuracy
study. Histograms and automatic segmentation can still fail on real SEM contrast.
Separate touching fibers, recover hidden fibers, and infer depth from a single SEM
remain ambiguous. Traced counts and apparent widths must not be used as validated
QC results without comparison to an independently annotated benchmark.

The 500× fields are especially vulnerable to lost fine fibers during tracing.
The 10,000× fields expose bright-rim/dark-core segmentation artifacts. Three-class
segmentation improves some visible masks, but no universal accuracy claim follows.
The default 6-pixel native-width exclusion is a heuristic, not a validated optical
resolution criterion. Systematic sampling still favors sites with visible edges.

## Proposed validation design

Define success separately for execution, crop/calibration, diameter agreement,
fiber detection/counting, and crossing order. Predefine permissible errors and
how abstentions count. Freeze an annotated test set that is never used for tuning.
Include independent specimens and acquisition sessions rather than treating
hundreds of correlated pixels on one fiber as independent trials.

Use multiple blinded analysts across repeated sessions and eventually sites.
Match identical sampling locations for edge-placement agreement, and separately
score independently selected locations for missed-fiber/selection bias. Evaluate
per-image count error, segmentation agreement, diameter bias/absolute error,
within-image and between-analyst variability, crossing accuracy, and abstention
rate. Maintain both raw images and all parameter/version manifests.

For perspective, if success were a genuinely independent binary image outcome,
zero failures in 299 representative independent cases gives a one-sided exact
95% lower bound just above 99% (0.05 ** (1/299)). Thirteen images—even perfect—
cannot support that claim. This illustration is not a sample-size prescription;
correlation, tolerances, and the intended claim determine the actual study design.

Kyle's DiameterJ suggestion is sound as a comparator, with method-specific
qualification. Its documented limits include segmentation dependence, touching
fibers, and resolution effects. Its prior validation does not establish the
performance of this implementation or a new laboratory's specimen workflow.
See the [official DiameterJ guidance](https://imagej.net/plugins/diameterj) and
[NIST's description](https://www.nist.gov/mml/bbd/biomaterials/diameterj).
The [Depth Anything V2 model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
describes a pretrained depth estimator; this review supplies no SEM depth labels
and therefore does not validate the model for this domain.
