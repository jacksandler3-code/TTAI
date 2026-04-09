# Table Tennis AI — System Specification

## What this is

A system that takes raw table tennis training footage and automatically detects rallies, removing dead time and producing highlights reels. Long-term goal is full game understanding: score tracking, shot classification, let detection.

Designed from the start to be deployed as a paid web service.

---

## What we learned building the heuristic version

Before committing to ML, we tried two vision-based approaches:

**Attempt 1 — MOG2 background subtraction + blob size filtering**
The idea: the ball creates small foreground blobs (30–2500 px²), people walking create large ones (>6000 px²). Filter by size to get a "ball activity" signal.
**Why it failed:** The ball is usually invisible in real footage — motion blur, overexposure, distance. The signal was too noisy to threshold reliably.

**Attempt 2 — Farneback dense optical flow within a central ROI**
The idea: measure how much the players and paddles are moving inside the central 70% of the frame. Rallies = sustained high flow. Dead time = near-zero flow.
**Why it's not reliable enough:** Walking, coaching, players warming up all generate flow. The adaptive threshold can't reliably separate "playing" from "moving around". Works on clean footage, breaks on busy gym environments.

**Conclusion:** Heuristics have a ceiling of ~70–80% accuracy and require per-video tuning. ML is the right path for a product.

---

## The ML approach

### Why VideoMAE

- **Table tennis is a Kinetics-400 class.** VideoMAE is pretrained on Kinetics-400 which includes table tennis. The model has already seen this domain — fine-tuning is adjusting an existing concept, not teaching from scratch.
- **Temporal understanding.** VideoMAE takes 16 frames from a clip and understands motion patterns over time, not just single-frame appearance. Critical for distinguishing "rally swing" from "walking through".
- **HuggingFace ecosystem.** Straightforward fine-tuning, well-documented, actively maintained.

**Model:** `MCG-NJU/videomae-base-finetuned-kinetics`

**For production inference** (once validated): `X3D-S` pretrained on Kinetics-400. ~7× faster than VideoMAE-base on CPU. Same fine-tuning process.

---

## Phase 1 — Rally detection

### Goal
Binary classifier: each 2-second window of footage = **rally** or **not rally**.

### Step 1 — Zero-shot baseline test

Before labeling anything, run the pretrained Kinetics model on your footage. Table tennis is already a class. If it gets 70–80% accuracy with zero training, you need far less labeled data.

```bash
python scripts/test_zero_shot.py training_session.mp4 --compare-optical-flow
```

This prints an ASCII timeline and score graph. It tells you whether the pretrained weights already work for your specific footage.

### Step 2 — Data collection

Footage requirements:
- Minimum 2–3 hours of labeled footage to start
- Multiple sessions (different days, lighting conditions)
- At least 2 different camera angles if possible
- Include edge cases: warm-up rallies, multi-ball training, coaching demonstrations

### Step 3 — Labeling with Label Studio

**Install:**
```bash
pip install label-studio
label-studio start
```
Runs locally at `http://localhost:8080`. Footage never leaves your machine.

**Project XML config** (create this as a new project in Label Studio):
```xml
<View>
  <Video name="video" value="$video" framerate="30"/>
  <VideoTimelineLabels name="label" toName="video">
    <Label value="rally" background="#00AA00"/>
    <Label value="not_rally" background="#AA0000"/>
  </VideoTimelineLabels>
</View>
```

**Workflow:**
1. Import video into Label Studio
2. Play through, drag timeline segments and label them "rally"
3. Anything unlabeled is implicitly "not rally" — only label the rally segments
4. Export as JSON

**Speed tip:** Pre-label first using the zero-shot model predictions, then just correct mistakes. Correction is 5–10× faster than labeling from scratch. The `scripts/prelabel.py` script generates a Label Studio import file from model predictions.

### Step 4 — Extract training clips

From the Label Studio export JSON, extract 2-second clips centred on each annotation boundary:

```bash
python scripts/export_clips.py annotations_export.json \
  --output-dir training_data/ \
  --clip-duration 2.0 \
  --negative-ratio 2   # 2 not_rally clips per rally clip
```

Output structure:
```
training_data/
  rally/        clip_001.mp4, clip_002.mp4, …
  not_rally/    clip_001.mp4, clip_002.mp4, …
```

Aim for at least 500 clips per class before fine-tuning. 1000+ per class is better.

### Step 5 — Fine-tune VideoMAE

Fine-tuning runs on Google Colab free tier (T4 GPU). Takes ~1 hour for 1000 clips.

```python
from transformers import (
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification,
    TrainingArguments, Trainer
)

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,
    id2label={0: "not_rally", 1: "rally"},
    label2id={"not_rally": 0, "rally": 1},
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="./rally-detector",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
```

### Step 6 — Replace the inference engine

The `core/pipeline.py` `analyze()` method currently calls `BallTracker` (optical flow). Replace it with a `VideoMAEAnalyzer` class that:
1. Slides a 2-second window with 0.5-second stride
2. Runs the fine-tuned model on each window
3. Returns the same `list[FrameSample]` interface
4. Feeds into the same `RallyDetector` signal processing (smoothing, thresholding, segmentation, padding)

The rest of the pipeline — FFmpeg export, CLI, SaaS architecture — is unchanged.

### Step 7 — Active learning loop

Each time you process new footage:
1. Run inference → pre-labels
2. Human reviews in Label Studio → corrects errors
3. Corrections added to training set
4. Re-fine-tune periodically

Each iteration the model improves and requires less human correction.

---

## Phase 2 — Score tracking

**Requires:** Ball tracking + point outcome detection.

### Ball tracking
Use **TrackNet** — published specifically for table tennis ball tracking, handles motion blur, available open-source. Fine-tune on your own footage for best results.

Once you have per-frame ball coordinates, most game events become geometric:

```
Ball trajectory → bounce detected → which side of table → point outcome
Ball exits frame boundary → ball out → point to opponent
Ball hits net, no crossing → net error → point to opponent
```

### Score logic (pure rules, no ML)
```python
class ScoreTracker:
    def on_point_won(self, winner: str):
        # Standard rules: first to 11, win by 2, serve changes every 2 points
        # Deuce at 10-10: serve changes every point
```

### Point outcome detection
Binary classifier per rally: "player 1 wins" or "player 2 wins". Train on labeled rally endings. Features: ball trajectory at end of rally, player position, body language.

---

## Phase 3 — Shot classification

**Requires:** Player pose estimation + ball contact detection.

- **Pose:** MediaPipe Pose or YOLO-Pose (free, runs on CPU, 33 landmarks per player)
- **Features:** Wrist velocity at contact, elbow angle, hip rotation
- **Classes:** forehand/backhand × topspin/backspin/sidespin/float + smash, push, flick, block

Minimum ~200 labeled examples per shot type. Multi-class fine-tune.

---

## Phase 4 — Let serve detection

**Very hard.** Requires:
- Ball tracked through the net plane with sub-centimetre accuracy
- Service box landing detection

Only viable with high-resolution footage (1080p minimum) and a camera angle that clearly shows the net. Probably Phase 4 or later.

---

## SaaS architecture

The `core/pipeline.py` `Pipeline` class is already designed for this. The web service only needs to:

1. Accept video upload → save to temp storage (S3/GCS)
2. Queue a processing job (Redis/SQS)
3. Worker calls `Pipeline(config).run(input_path, output_path, progress_callback=ws_send)`
4. Upload output to cloud storage
5. Notify user via WebSocket

```python
# Future FastAPI endpoint — core logic unchanged
@app.post("/process")
async def process(file: UploadFile, config: PipelineConfig = Body(...)):
    tmp_in  = await save_to_temp(file)
    tmp_out = tmp_in.with_suffix("_highlights.mp4")
    result  = Pipeline(config).run(str(tmp_in), str(tmp_out),
                                   progress_callback=lambda p: ws.send(p))
    url = await upload_to_s3(tmp_out)
    return {"url": url, **result.to_dict()}
```

**Pricing model suggestion:**
- Free tier: 10 minutes of footage/month
- Starter: £9/month — 5 hours/month
- Club: £29/month — unlimited, score tracking when available
- Academy: custom pricing, shot analytics, API access

---

## Technical stack

| Layer | Technology |
|---|---|
| Video processing | OpenCV, FFmpeg |
| ML inference | PyTorch, HuggingFace Transformers |
| Base model | VideoMAE-base (Kinetics-400) |
| Production model | X3D-S (Kinetics-400) |
| Labeling | Label Studio (self-hosted) |
| Ball tracking | TrackNet (phase 2) |
| Pose estimation | MediaPipe Pose (phase 3) |
| Web API | FastAPI |
| Job queue | Redis + Celery or SQS + Lambda |
| Storage | S3 or GCS |
| Frontend | TBD |

---

## File structure (target)

```
core/
  pipeline.py          Orchestrator — public interface for API and CLI
  motion_analyzer.py   VideoMAE sliding window inference (replaces ball_tracker)
  rally_detector.py    Signal processing: smooth → threshold → segment → pad
  video_exporter.py    FFmpeg: highlights / individual clips / speedup
  ball_tracker.py      TrackNet wrapper (phase 2)
  score_tracker.py     Rules-based score logic (phase 2)

scripts/
  test_zero_shot.py    Test pretrained VideoMAE on footage before labeling
  prelabel.py          Generate Label Studio import JSON from model predictions
  export_clips.py      Extract training clips from Label Studio export

label_studio_ml/
  server.py            Flask ML backend — Label Studio calls this to pre-label
  label_config.xml     Label Studio project configuration XML

training/
  train.py             Fine-tune VideoMAE on extracted clips
  evaluate.py          Confusion matrix, per-class F1, timeline accuracy

utils/
  video_utils.py       Video metadata helpers

main.py                CLI entry point
requirements.txt       Core deps (opencv, numpy)
requirements_ml.txt    ML deps (transformers, torch)
```

---

## Immediate next steps

1. **Run zero-shot test** on 10–15 minutes of your footage. If accuracy looks reasonable (roughly right segments), proceed. If it's completely wrong, check camera angle and lighting — the model might need more domain-specific fine-tuning.

2. **Set up Label Studio locally.** Import one session of footage. Spend 30 minutes labeling rally segments. This gives you a feel for the tool and a first small dataset.

3. **Build `scripts/prelabel.py`** — takes zero-shot predictions + optical flow, outputs Label Studio JSON. This makes all future labeling sessions faster.

4. **Collect and label 2–3 hours of footage.** This is the highest-leverage work right now. More labeled data beats better model architecture.

5. **Fine-tune on Colab.** Free T4 GPU, takes about 1 hour.

6. **Integrate the fine-tuned model** into `core/pipeline.py` as a drop-in replacement for the optical flow analyzer.
