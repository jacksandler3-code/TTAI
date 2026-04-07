# Table Tennis Rally Detector

Turns a raw training session recording into a highlights reel with one command.
No ML models. No manual table selection. No GPU required.

```
python main.py training_session.mp4
# → training_session_highlights.mp4
```

---

## How it works

**Traditional approach (fragile):** measure average motion in a region → walking through frame = false positive.

**This approach:** background subtraction + blob size filtering.

- Large foreground blobs (≥ 6,000 px²) = players/people → **ignored**
- Small foreground blobs (30–2,500 px²) = ball or paddle tip → **rally signal**

People walking through frame create large blobs and are silently discarded.
The ball and paddle tips during active play create small, fast-moving blobs.
No knowledge of where the table is required.

---

## Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and available in `PATH`
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from https://ffmpeg.org/download.html

---

## Installation

```bash
git clone <repo>
cd table-tennis-detector
pip install -r requirements.txt
```

---

## Usage

### Fully hands-free (default)

```bash
python main.py training.mp4
```

Produces `training_highlights.mp4` in the same directory. No prompts.

### Export modes

```bash
# Highlights reel (default) — all rallies joined into one file
python main.py training.mp4 --export highlights

# Individual clips — rally_001.mp4, rally_002.mp4, …
python main.py training.mp4 --export individual

# Speed-up version — full video, dead time at 8× speed
python main.py training.mp4 --export speedup
```

### Analysis only (no export)

```bash
python main.py training.mp4 --analyze-only
```

Prints the detected rally timestamps without writing any video.

### Tuning parameters

| Flag | Default | Notes |
|---|---|---|
| `--sensitivity` | `0.5` | Lower = more sensitive. Try `0.3` if rallies are missed. |
| `--min-duration` | `2.0` s | Drop very short detections. |
| `--merge-gap` | `2.0` s | Merge rallies separated by a brief pause. |
| `--sample-rate` | `3` | Process every Nth frame. Higher = faster, less precise. |
| `--speedup-factor` | `8.0` | Speed multiplier for dead time in speedup mode. |

```bash
# More sensitive — catches short exchanges and serve sequences
python main.py training.mp4 --sensitivity 0.3

# Faster processing on a slow machine
python main.py training.mp4 --sample-rate 6

# Custom output location
python main.py training.mp4 -o /path/to/output.mp4
```

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

Or run the standalone script directly (no pytest required):

```bash
python tests/test_synthetic.py
```

This generates a synthetic video with known rally periods, runs the full
pipeline, and verifies each ground-truth rally is detected within ±2 s.

---

## Project structure

```
main.py                  CLI entry point
core/
  pipeline.py            Orchestrator — the single public interface
  ball_tracker.py        MOG2 background subtraction + blob filtering
  rally_detector.py      Signal smoothing, thresholding, segmentation
  video_exporter.py      FFmpeg-based export (highlights / individual / speedup)
utils/
  video_utils.py         Video metadata helpers
tests/
  test_synthetic.py      Synthetic end-to-end test
requirements.txt
```

---

## Troubleshooting

**No rallies detected**
Lower `--sensitivity`: `python main.py video.mp4 --sensitivity 0.3`

**Too many false positives**
Raise `--sensitivity` or increase `--min-duration`:
`python main.py video.mp4 --sensitivity 0.8 --min-duration 3`

**Rallies split in the middle (let ball / brief pause)**
Increase `--merge-gap`: `python main.py video.mp4 --merge-gap 3`

**Processing is slow**
Increase `--sample-rate` (e.g. `--sample-rate 6`).  At 30 fps this samples
5 frames/second, which is more than enough temporal resolution.

**FFmpeg error on export**
Make sure `ffmpeg` is in your PATH: run `ffmpeg -version` to confirm.

**Corrupted / unreadable video**
Try re-muxing first: `ffmpeg -i input.mp4 -c copy fixed.mp4`

---

## SaaS / web service notes

The `Pipeline` class in `core/pipeline.py` is deliberately decoupled from the
CLI and any GUI.  It takes a local file path, an optional progress callback,
and returns a plain data object.  Wrapping it in a web service requires only:

1. Save the uploaded file to a temp path.
2. Call `Pipeline(config).run(tmp_path, out_path, progress_callback=send_ws)`.
3. Upload the output to cloud storage.
4. Return the download URL.

No changes to the detection or export logic are needed.

Example FastAPI sketch:

```python
from fastapi import FastAPI, UploadFile
from core.pipeline import Pipeline, PipelineConfig

app = FastAPI()

@app.post("/process")
async def process(file: UploadFile):
    tmp_in  = save_to_temp(file)
    tmp_out = tmp_in.with_suffix("_highlights.mp4")
    result  = Pipeline().run(str(tmp_in), str(tmp_out))
    url     = upload_to_s3(tmp_out)
    return {"url": url, **result.to_dict()}
```
