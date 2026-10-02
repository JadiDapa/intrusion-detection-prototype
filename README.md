# Restricted Zone Intrusion Detection

YOLO26 + ByteTrack + polygon zone. A person is flagged as an intruder when the
bottom-center of their bounding box (their feet) stays inside the zone for at
least `--dwell` seconds. Each tracked person alerts at most once per `--cooldown`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

NVIDIA GPU on your laptop? Install the CUDA build of PyTorch first
(https://pytorch.org/get-started/locally/), then `pip install -r requirements.txt`.

## Web UI (Streamlit)

```bash
streamlit run app.py
```

- **Sidebar**: video source (defaults to `street.mp4`; upload a file, or enter a path / RTSP URL / webcam index),
  model, classes, confidence, dwell/cooldown, and "process every Nth frame" (default 2 to speed up CPU runs).
- **① Zone**: the polygon starts from `zone.json`. Click on the frame to add corners, Undo/Clear/Reset, and
  **Save** writes it back to `zone.json` (used by `detect.py` too). You can also load or download a zone JSON.
- **② Detect & dashboard**: live annotated preview while it runs, then KPIs, a people-over-time chart with
  intrusion periods, alerts over time, time inside the zone per person, a foot-traffic heatmap, confidence
  distribution, the alert log with snapshots, the annotated video, and CSV downloads.

Each run is saved to `events/runs/<timestamp>/` (`events.csv`, `snapshots/`, `output.mp4`).
Changing a setting while a run is in progress stops it, and the results so far are kept.

## 1. Draw the zone

```bash
python draw_zone.py --source 0              # webcam
python draw_zone.py --source my_video.mp4   # video file
```
Left click adds points, right click / `u` undoes, `s` saves to `zone.json`.

## 2. Run detection

```bash
python detect.py --source 0 --zone zone.json
python detect.py --source my_video.mp4 --zone zone.json --save-video
```

Box colors: green = outside zone, orange = inside but not yet confirmed,
red = confirmed intruder. Press `q` to quit.

Outputs go to `events/`: `events.csv` (log), `snapshots/` (one image per alert),
and the annotated video if `--save-video` is set.

## Useful options

| Option | Default | Notes |
|---|---|---|
| `--model` | `yolo26n.pt` | Try `yolo26s.pt` for better accuracy |
| `--classes` | `person` | Which classes count as intruders: names or ids, comma-separated, or `all` |
| `--conf` | `0.4` | Raise to cut false alarms, lower to catch distant people |
| `--imgsz` | `640` | Raise (e.g. 960, 1280) for small/distant people; slower |
| `--dwell` | `1.0` | Seconds inside before alerting |
| `--cooldown` | `30` | Seconds before the same person alerts again |
| `--tracker` | `bytetrack.yaml` | Or `botsort.yaml` (better ID re-identification, slower) |
| `--device` | auto | `cpu`, `0` (first GPU) |

## Training your own model

Open `train_intrusion_yolo26.ipynb` in Google Colab (GPU runtime), paste your
Roboflow export link, and run all cells. Download `best.pt`, put it here, then:

```bash
python detect.py --source 0 --zone zone.json --model best.pt --classes human
```

## Next steps
- Notifications: fill in the `TODO` in `detect.py` (Telegram, siren, dashboard)
- Test on real footage from the target cameras (day, night/IR, rain)
- Export for speed: `yolo export model=yolo26n.pt format=onnx` (or `engine` for TensorRT)
