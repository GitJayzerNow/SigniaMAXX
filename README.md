# ASL Fingerspelling Recognizer

Real-time, fully offline recognition of ASL fingerspelling — the letters
A-Z and digits 0-9 only (no words/sentences). Your webcam feed is
processed entirely on your machine: hand landmarks are extracted in the
browser with MediaPipe, and a locally-trained classifier (running in
Flask) turns those landmarks into a predicted letter/number.

```
Browser (Next.js)                Backend (Flask)
┌─────────────────────┐          ┌──────────────────────┐
│ Webcam → MediaPipe   │  WebSocket │ Landmark → Classifier │
│ Hands (21 landmarks) │ ───────► │ → predicted label      │
│ Skeleton overlay     │ ◄─────── │                        │
└─────────────────────┘          └──────────────────────┘
```

Landmarks (not raw images) are sent over the socket — this keeps things
fast, lightweight, and works well on CPU only.

## Why landmarks instead of image classification?

A CNN trained on raw webcam images needs a GPU to run at real-time FPS
and is very sensitive to lighting/background/skin tone variance. Hand
*landmark* coordinates (21 3D points per hand) normalize all of that
away — the classifier just learns finger geometry, so it's smaller,
faster, and trains from far fewer examples.

## Project structure

```
asl-recognizer/
├── backend/
│   ├── server.py            # Flask + SocketIO inference server (GPU via CUDA)
│   ├── collect_data.py      # (Optional) webcam tool to record your own samples
│   ├── convert_dataset.py    # Converts a public image dataset to landmarks (no recording needed)
│   ├── convert_npy_dataset.py # Converts an X.npy/Y.npy dataset (e.g. digit datasets) to landmarks
│   ├── train.py              # Trains PyTorch MLP from data/landmarks.csv
│   ├── model_def.py          # Shared PyTorch model definition
│   ├── data/                # landmarks.csv lives here after collection
│   └── model/                # classifier.pt lives here after training
├── frontend/
│   ├── app/                  # Next.js app (page, layout, styles)
│   ├── components/HandCanvas.js  # MediaPipe Hands wrapper
│   ├── scripts/download-mediapipe-assets.sh  # one-time offline asset fetch
│   └── public/mediapipe/hands/   # model files live here after download
├── requirements.txt
└── README.md
```

## 1. Setup

### Backend (Python)

```bash
cd asl-recognizer
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install PyTorch with CUDA support separately, so it picks up your GPU
# (RTX 5050 needs CUDA 12.4+ wheels — the default `pip install torch`
# can silently give you a CPU-only or mismatched build):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify the GPU is detected:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True  NVIDIA GeForce RTX 5050
```

If `torch.cuda.is_available()` prints `False`, your NVIDIA driver is
likely older than what CUDA 12.4 needs — update it from
[nvidia.com/drivers](https://www.nvidia.com/drivers) and re-check.
Both `train.py` and `server.py` auto-detect CUDA and fall back to CPU
if it isn't available, so the app still works either way.

### Frontend (Node.js 18+)

```bash
cd frontend
npm install
```

### Make it work fully offline

MediaPipe's hand-tracking model normally loads its `.wasm`/model files
from a CDN the first time. To make the whole app work with **no
internet at all** (after first setup), fetch those files once and
serve them locally:

```bash
cd frontend
bash scripts/download-mediapipe-assets.sh
```

**Windows (PowerShell) users:** `bash` isn't available by default in
PowerShell. Use the PowerShell version instead:

```powershell
cd frontend
.\scripts\download-mediapipe-assets.ps1
```

If PowerShell blocks the script from running (execution policy error),
run this once first, then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This saves the files into `frontend/public/mediapipe/hands/`, which the
app is already configured to load from (`HandCanvas.js` points
`locateFile` at `/mediapipe/hands/...`). After this, disconnect from the
internet — it'll still work.

## 2. Get your dataset

You have two options — pick one.

### Option A: Convert a public dataset (recommended, no manual signing)

Fully automated — no camera, no recording, no button presses. Download
a public ASL image dataset such as the
[Kaggle "ASL Alphabet" dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
(~87k images, A-Z), unzip it, then run:

```bash
cd backend
python convert_dataset.py /path/to/dataset_root
```

The dataset should have one subfolder per letter/number
(`A/`, `B/`, ... `0/`, `1/`, ...) each full of images — this matches
the standard Kaggle ASL Alphabet layout. The script runs MediaPipe over
every image unattended, extracts landmarks, and writes
`backend/data/landmarks.csv`. Images where no hand is detected are
skipped automatically (it'll print a per-class count).

Useful flags:

```bash
# Cap images per class (faster conversion, still plenty of data)
python convert_dataset.py /path/to/dataset_root --max-per-class 500

# Combine with an existing landmarks.csv instead of overwriting it
python convert_dataset.py /path/to/dataset_root --append
```

Note: since the dataset was recorded on someone else's camera/lighting/
hands, live accuracy on *your* webcam may be somewhat lower than a
self-recorded set. If accuracy feels off after training, you can
supplement with a small batch of your own samples (Option B) using
`--append` combined with `collect_data.py`.

### Digits dataset in .npy format

Some public digit datasets (e.g. the ardamavi "Sign Language Digits
Dataset") ship as `X.npy` + `Y.npy` arrays instead of image folders.
Use the separate converter for this format:

```bash
cd backend
python convert_npy_dataset.py /path/to/X.npy /path/to/Y.npy --append
```

`--append` adds these to your existing `landmarks.csv` (e.g. after
converting a letters dataset) instead of overwriting it.

Two things to check before trusting the output:

- **Label order**: this specific dataset's one-hot columns are
  ordered `9,0,1,2,3,4,5,6,7,8`, not `0-9` — the script defaults to
  that, but verify against your dataset's own page/README and pass
  `--label-order 0,1,2,3,4,5,6,7,8,9` if yours is normal order.
- **Skip rate**: these images are often only 64x64 and tightly
  cropped, which can make hand landmark detection fail on a large
  fraction of them (the script upscales automatically, but this only
  helps so much). Check the printed skip count after running — if
  most images are being skipped, this dataset may not convert well
  and self-recording digits with `collect_data.py` may work better.

### Option B: Record your own samples (more accurate, more effort)

```bash
cd backend
python collect_data.py
```

- Press a key (A-Z or 0-9) to select which sign you're about to record.
- Hold **SPACE** while showing that sign to the camera — it records a
  landmark sample every frame while held.
- Move your hand slightly (angle, distance, rotation) between bursts for
  variety — aim for **150-300+ samples per class**.
- Press **c** to clear the current label, **q** to quit and save.

This appends to `backend/data/landmarks.csv`.

> Note: standard ASL fingerspelling has motion for J and Z. Since this
> system only reads static poses, treat J/Z as their most distinct
> static hand-shape approximation, or exclude them from your dataset
> and drop them from `LABELS` in `collect_data.py`.

## 3. Train the classifier

```bash
cd backend
python train.py
```

This trains a small PyTorch MLP on your CSV and saves
`backend/model/classifier.pt`, along with a per-class accuracy report
printed to the console. It prints which device it's training on —
confirm it says `cuda` (and your GPU name) rather than `cpu`. Re-run
this any time you add more data. Training this small a model is fast
either way, but inference in `server.py` also runs on GPU, keeping the
CPU free for MediaPipe hand tracking and the webcam feed.

## 4. Run the app

Two terminals:

```bash
# Terminal 1 — backend
cd backend
python server.py
# → running on http://localhost:5000

# Terminal 2 — frontend
cd frontend
npm run dev
# → running on http://localhost:3000
```

Open `http://localhost:3000`, allow camera access, and show a sign —
the letter/number and confidence appear live. Holding a sign steady for
~0.7s auto-appends it to the "Spelled output" buffer below.

## Improving accuracy

- Collect more samples, especially for signs the report shows as weak.
- Collect data in a few different lighting conditions / backgrounds.
- Keep one hand only, centered, and reasonably close to camera for both
  data collection and live use (matches training distribution).
- If you want more capacity, widen the layers in `model_def.py`
  (`LandmarkClassifier`) — the feature format (63-dim normalized
  landmark vector) stays the same either way.

## Environment variable (optional)

If you run the backend on a different host/port, set this before
`npm run dev`:

```bash
NEXT_PUBLIC_BACKEND_URL=http://your-host:5000
```

## Scope

This system intentionally recognizes **only static single letters
(A-Z) and digits (0-9)** — no words, sentences, or motion-based signs.
It's built as a fingerspelling helper, not a full ASL translator.
