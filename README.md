# SwingTool

Computer-vision analysis of golf swings from a single phone video. Current
vertical slice: **video → person detection → ViTPose keypoints → skeleton
overlay + JSON**.

All models are Apache-2.0 (commercial-clean):

| Stage | Model | License |
|---|---|---|
| Person detection | [`PekingU/rtdetr_r50vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365) | Apache-2.0 |
| Pose estimation | [`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) | Apache-2.0 |

## Setup (Windows / PowerShell)

Requires Python 3.10+ and an NVIDIA GPU with a CUDA 12.x driver.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# IMPORTANT: torch must come from the CUDA index. A plain `pip install torch`
# silently installs the CPU-only build.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify CUDA before running anything:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True`. If it prints `False`, the CPU
wheel sneaked in — reinstall torch with the `--index-url` line above.

## Run

```powershell
python -m swingtool analyze samples\swing.mov
```

Outputs land in `output/`:

- `keypoints.json` — per-frame COCO-17 keypoints with confidences
  (schema in `swingtool/schema.py`; coordinates are always original-resolution
  pixels, regardless of `--max-dim`)
- `overlay.mp4` — the input video with the tracked skeleton drawn on

Useful flags:

| Flag | Effect |
|---|---|
| `--frame-stride N` | Process every Nth frame (faster, fewer samples) |
| `--max-dim N` | Downscale for inference so the longest side ≤ N (saves VRAM/time; output coords are still original-resolution) |
| `--device cpu` | Deliberate CPU fallback. The default (`cuda`) errors out if CUDA is missing rather than silently running on CPU |
| `--overlay-min-score S` | Hide keypoints below confidence S in the overlay |

The first run downloads ~500MB of model weights to the Hugging Face cache.

## Footage notes

- **Portrait phone clips work.** Rotation metadata is honored; the pipeline
  treats the decoded frame's shape as authoritative.
- **Filmed from behind:** face keypoints (nose/eyes/ears) will have low
  confidence and are hidden in the overlay by default. Keypoint left/right is
  **anatomical** — from behind, the golfer's left shoulder appears on the
  left of the image (this is the mirror of a front-facing view; downstream
  metric code must not flip it).

## Phase 2 — 2D swing metrics

```powershell
python -m swingtool metrics output\keypoints.json
```

Writes `output/metrics.json` (schema in `swingtool/metrics/schema.py`) and
prints a summary: swing events (address/top/impact), tempo, head stability,
knee flex, spine tilt. Every value carries a `quality` flag
(`reliable` / `view_dependent` / `approximate_2d` / `low_confidence`) — no
faked 3D.

## Phase 3A — club/ball detection and club-path metrics

Zero-shot detection with Grounding DINO (`IDEA-Research/grounding-dino-tiny`,
Apache-2.0), then pure-geometry derivation, then the overlay:

```powershell
python -m swingtool detect samples\swing.mov output\keypoints.json   # -> detections.json
python -m swingtool derive output\keypoints.json output\detections.json  # -> analysis.json
python -m swingtool render-club samples\swing.mov output\analysis.json --keypoints output\keypoints.json
```

`analysis.json` (schema in `swingtool/analysis/schema.py`) holds the club-path
trace, relative club speed, and ball position — each honestly flagged.

## Phase 3B — depth (swing plane + X-factor)

Monocular relative depth with Depth Anything V2 **Small**
(`depth-anything/Depth-Anything-V2-Small-hf`, Apache-2.0), then re-derive with
depth to add the depth-assisted metrics:

```powershell
python -m swingtool depth samples\swing.mov output\keypoints.json output\detections.json  # -> depth_samples.json
python -m swingtool derive output\keypoints.json output\detections.json --depth output\depth_samples.json
```

This adds `depth_assisted` to `analysis.json`: **swing-plane tilt** (how far the
swing plane comes out of the 2D image) and **X-factor** (hip/shoulder
separation at the top). Both are labeled `depth_assisted_approximate`.

**Honesty:** Depth Anything outputs **relative, scale-free** depth — never
metric distance. So swing plane and X-factor are *approximate orientation*, not
true 3D, and are flagged as such. Only the **Small** checkpoint is used
(Base/Large/Giant are CC-BY-NC). Depth runs on GPU (fp16); the low-res
prediction is upscaled on CPU (a large GPU interpolate faults on this build).

**Honesty constraints baked in (these matter):**
- Depth Anything / this pipeline give **relative**, not metric, information.
  Club speed is reported in **body-lengths per second** (`relative_only`,
  `coarse`) — **never mph/km·h⁻¹/m·s⁻¹**. The ~7-frame downswing at 30fps is
  badly undersampled, so speed is a coarse window estimate, not a precise number.
- **Gaps are never fabricated.** When detection fails (motion blur through the
  downswing), the club point is `null`; only short gaps (≤3 frames) are
  interpolated and flagged `interpolated`.
- **Detection runs on CPU by default.** Grounding DINO's deformable-attention
  `grid_sample` hits an illegal-memory-access on this torch/CUDA build; CPU is
  correct and deterministic. `--device cuda` remains available. To stay
  tractable (~15–18s/frame), detection is auto-scoped to the swing window via
  the Phase-2 event detector.

## Phase 3C — ball flight & shot shape (predicted)

`derive` also produces a `ball_flight` block, and `render-club` draws a
broadcast-style tracer plus an end-card naming the shot
(**slice / hook / fade / draw / straight**):

```powershell
python -m swingtool render-club samples\swing.mov output\analysis.json --keypoints output\keypoints.json
```

- **Solid magenta** = the ball where it was *actually detected* after impact.
  Usually 0–3 points — the ball leaves frame in ~1–2 blurred frames — and a
  static tee/range ball is explicitly rejected (it isn't flight).
- **Dashed orange** = a **modelled** predicted arc. Drawn dashed on purpose so
  it never reads as a tracked ball. Image-space only; no distance/scale.
- **End-card** = the predicted shot shape, held for `--endcard-seconds` (2.5 by
  default; `0` disables).

**Honesty — read this before showing it to anyone.** Shot shape is driven by
**club face-to-path angle and spin**, which a launch monitor measures directly
and which this pipeline does **not** measure. The label is a *model estimate*
from weak 2D proxies:

- start direction (≈ face) from the first displaced post-impact ball, when we
  get one;
- club-path lean from the 2D club-head trace through impact;
- shape from `start − path` via the standard ball-flight relationship, with
  our own coarse curvature tolerances (`tolerance_ours`).

Every value is flagged `model_estimate` with low confidence, and the end-card
says on-screen that curvature/spin are not measured and to confirm with a
launch monitor. When the ball is never seen in flight, the arc falls back to a
square-face prior (shape from path alone) at even lower confidence — clearly
flagged. **This is a visual estimate for demos, not a launch-monitor reading.**

## Phase 4 — swing report

```powershell
python -m swingtool report output\metrics.json
```

Compares the measured metrics against **cited** reference ranges and writes
`output/report.json` plus a readable summary: highlights first, findings
("worth a look") with the source for every threshold, a "measured, not judged"
section for everything we refuse to threshold, fun facts, and limitations.

- **Deterministic rules engine — no LLM decides anything.** Every finding
  traces to a measured metric, a threshold from the versioned
  `swingtool/report/references_v1.json`, and a citation (Novosel & Garrity
  2004 for tempo; Hume, Keogh & Reid 2005 for lead-leg extension; McLean 1992 /
  Myers et al. 2008 for X-factor). Numbers that are this project's tolerances
  rather than the source's are flagged `tolerance_ours`.
- **Quality flags gate findings.** `approximate_2d`, `depth_assisted_approximate`,
  relative-only, and low-confidence metrics are suppressed (visibly, with the
  reason) or hedged — never turned into confident sentences. Unsourceable
  thresholds (head drift, spine tilt) are left out rather than guessed.
- **Self-comparison:** each run appends to `output/history.jsonl`; later runs
  open with progress vs your own last swing (`--no-history` to skip).
- This is observations from **one swing, one camera, 30fps** — not a diagnosis,
  and **not a substitute for a qualified coach or a launch monitor**. The
  report says so itself.

## Tests

```powershell
python -m pytest
```

Tests cover streaming/stride logic, coordinate mapping, the schema contracts,
2D-metric geometry, event detection on low-confidence gaps, and Phase-3
club-path gap handling (asserting gaps are flagged, not fabricated, and that
no physical-scale units appear). No GPU or model downloads needed.

## Layout

```
swingtool/
  cli.py        argparse + dispatch only
  schema.py     Phase-1 pose contract (AnalysisResult)
  ingest.py     streaming frame reader (never a whole clip in memory)
  pose/         person detection (RT-DETR) + keypoints (ViTPose)
  metrics/      Phase-2 2D swing metrics (pure geometry)
  detect/       Phase-3A club/ball detection (Grounding DINO, CPU)
  depth/        Phase-3B monocular relative depth (Depth Anything V2 Small, GPU)
  analysis/     Phase-3 derivation: club-path, speed, ball, swing-plane, X-factor,
                ball-flight prediction + shot-shape estimate (ballflight.py)
  report/       Phase-4 deterministic report engine + cited reference config
  render.py     skeleton + club-path/ball overlays, streamed to disk
```

Models are Apache-2.0 only. Each model stage loads, runs, and frees VRAM
before the next; depth and detection are never resident simultaneously.
Nothing derived from monocular depth claims a physical scale.
