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

## Tests

```powershell
python -m pytest
```

Tests cover streaming/stride logic, downscale coordinate mapping, the schema
contract, and rendering — no GPU or model downloads needed.

## Layout

```
swingtool/
  cli.py        argparse + dispatch only
  config.py     run configuration, device resolution
  schema.py     the pipeline contract (AnalysisResult)
  ingest.py     streaming frame reader (never a whole clip in memory)
  pose/         detection (RT-DETR) + keypoints (ViTPose), VRAM freed at stage end
  render.py     skeleton overlay, streamed to disk
```

Future stages (depth, club detection, metrics, API) plug in after the pose
stage and consume `AnalysisResult`.
