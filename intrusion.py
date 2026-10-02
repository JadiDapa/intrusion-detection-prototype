"""
Shared intrusion-detection logic used by detect.py (CLI) and app.py (Streamlit).

A person counts as an intruder when the bottom-center of their bounding box
(roughly their feet) stays inside the zone for at least `dwell` seconds.
Each tracked person triggers at most one alert per `cooldown` seconds.
"""
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")

# BGR colors used for drawing
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

CSV_HEADER = ["timestamp", "source_time_s", "track_id", "class", "confidence",
              "x1", "y1", "x2", "y2", "snapshot"]


def parse_source(source: str):
    """Return (cv2-ready source, is_live) for a webcam index, file path or stream URL."""
    src = int(source) if str(source).isdigit() else source
    is_live = isinstance(src, int) or str(src).lower().startswith(LIVE_PREFIXES)
    return src, is_live


def resolve_classes(names: dict, spec: str):
    """Turn 'person,helmet' or '0,2' or 'all' into class ids for a model's names dict."""
    if spec.strip().lower() in ("all", "*"):
        return None
    lookup = {str(v).lower(): int(k) for k, v in names.items()}
    ids = []
    for item in spec.split(","):
        item = item.strip()
        if item.isdigit() and int(item) in names:
            ids.append(int(item))
        elif item.lower() in lookup:
            ids.append(lookup[item.lower()])
        else:
            raise ValueError(f"Class '{item}' not found. This model's classes are: {list(names.values())}")
    return ids


def scale_zone(data: dict, frame_w: int, frame_h: int) -> np.ndarray:
    """Rescale a zone dict's polygon if the stream resolution differs from when it was drawn."""
    pts = np.array(data["polygon"], dtype=np.float32)
    sx = frame_w / data.get("frame_width", frame_w)
    sy = frame_h / data.get("frame_height", frame_h)
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts.astype(np.int32)


def load_zone(path: str, frame_w: int, frame_h: int) -> np.ndarray:
    with open(path) as f:
        return scale_zone(json.load(f), frame_w, frame_h)


@dataclass
class Detection:
    track_id: int
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2
    foot: tuple  # x, y
    inside: bool
    dwell: float
    confirmed: bool


@dataclass
class Alert:
    time: datetime
    source_time: float
    det: Detection
    snapshot: np.ndarray  # frame with zone + intruder box drawn


@dataclass
class FrameResult:
    frame: np.ndarray  # annotated
    detections: list = field(default_factory=list)
    alerts: list = field(default_factory=list)

    @property
    def intruders(self) -> int:
        return sum(d.confirmed for d in self.detections)

    @property
    def in_zone(self) -> int:
        return sum(d.inside for d in self.detections)


class ZoneMonitor:
    """Runs tracking on frames and applies the dwell/cooldown intrusion rules."""

    def __init__(self, model, zone: np.ndarray, class_ids=None, conf=0.4, imgsz=640,
                 tracker="bytetrack.yaml", dwell=1.0, cooldown=30.0, device=None):
        self.model = model
        self.zone = zone
        self.class_ids = class_ids
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.dwell = dwell
        self.cooldown = cooldown
        self.device = device
        self.entered_at: dict[int, float] = {}   # track_id -> time it entered the zone
        self.last_alert: dict[int, float] = {}   # track_id -> time of its last alert
        self.last_seen: dict[int, float] = {}    # track_id -> last time it was detected

    def update(self, frame: np.ndarray, now: float) -> FrameResult:
        """Track one frame. `now` is seconds (video time for files, wall time for live)."""
        result = self.model.track(frame, persist=True, classes=self.class_ids, conf=self.conf,
                                  imgsz=self.imgsz, tracker=self.tracker, device=self.device,
                                  verbose=False)[0]
        out = FrameResult(frame=frame)
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.int().cpu().numpy()
            for (x1, y1, x2, y2), tid, c, k in zip(xyxy, ids, confs, clss):
                tid = int(tid)
                self.last_seen[tid] = now
                foot = (float((x1 + x2) / 2), float(y2))
                inside = cv2.pointPolygonTest(self.zone, foot, False) >= 0
                if inside:
                    self.entered_at.setdefault(tid, now)
                    dwell = now - self.entered_at[tid]
                    confirmed = dwell >= self.dwell
                else:
                    self.entered_at.pop(tid, None)
                    dwell, confirmed = 0.0, False
                det = Detection(tid, self.model.names[int(k)], float(c),
                                (int(x1), int(y1), int(x2), int(y2)), foot, inside, dwell, confirmed)
                out.detections.append(det)

                if confirmed and now - self.last_alert.get(tid, -1e9) >= self.cooldown:
                    self.last_alert[tid] = now
                    snap = frame.copy()
                    cv2.polylines(snap, [self.zone], True, RED, 2)
                    cv2.rectangle(snap, det.box[:2], det.box[2:], RED, 3)
                    out.alerts.append(Alert(datetime.now(), now, det, snap))

        # Forget tracks not seen for a while so memory doesn't grow forever
        for tid in [t for t, s in self.last_seen.items() if now - s > 10]:
            for d in (self.entered_at, self.last_alert, self.last_seen):
                d.pop(tid, None)

        out.frame = self.draw(frame, out.detections)
        return out

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for d in detections:
            color = RED if d.confirmed else ORANGE if d.inside else GREEN
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(d.foot[0]), int(d.foot[1])), 5, color, -1)
            label = f"{d.cls} {d.track_id} {d.conf:.2f}" + (f" {d.dwell:.1f}s" if d.inside else "")
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        intruders = sum(d.confirmed for d in detections)
        zone_color = RED if intruders else GREEN
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone], zone_color)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.polylines(frame, [self.zone], True, zone_color, 2)
        if intruders:
            cv2.putText(frame, f"INTRUSION: {intruders}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, RED, 3)
        return frame


class EventLog:
    """Appends alerts to <dir>/events.csv and saves a snapshot per alert in <dir>/snapshots/."""

    def __init__(self, events_dir):
        self.dir = Path(events_dir)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.csv"
        new = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._csv = csv.writer(self._file)
        if new:
            self._csv.writerow(CSV_HEADER)

    def write(self, alert: Alert) -> Path:
        d = alert.det
        snap = self.dir / "snapshots" / f"{alert.time:%Y%m%d_%H%M%S}_id{d.track_id}.jpg"
        cv2.imwrite(str(snap), alert.snapshot)
        self._csv.writerow([alert.time.isoformat(timespec="seconds"), f"{alert.source_time:.2f}",
                            d.track_id, d.cls, f"{d.conf:.2f}", *d.box, str(snap)])
        self._file.flush()
        return snap

    def close(self):
        self._file.close()
