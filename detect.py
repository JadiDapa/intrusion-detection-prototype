"""
Restricted-zone intrusion detector: YOLO26 + ByteTrack + polygon zone.
Works with the pretrained COCO model or your own trained weights (--model best.pt).

A person counts as an intruder when the bottom-center of their bounding box
(roughly their feet) stays inside the zone for at least --dwell seconds.
Each tracked person triggers at most one alert per --cooldown seconds.

Usage:
    python detect.py --source 0 --zone zone.json
    python detect.py --source videos/test.mp4 --zone zone.json --save-video
    python detect.py --source rtsp://user:pass@ip/stream --zone zone.json --no-show
    python detect.py --source 0 --model best.pt --classes all      # your trained model

For a web UI with zone editing and charts: streamlit run app.py
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from intrusion import EventLog, ZoneMonitor, load_zone, parse_source, resolve_classes


def main():
    ap = argparse.ArgumentParser(description="Restricted zone intrusion detector")
    ap.add_argument("--source", default="0", help="webcam index, video path, or RTSP URL")
    ap.add_argument("--zone", default="zone.json", help="polygon JSON from draw_zone.py")
    ap.add_argument("--model", default="yolo26n.pt", help="YOLO weights (downloads automatically)")
    ap.add_argument("--classes", default="person",
                    help="classes that count as intruders: names or ids, comma-separated, or 'all'")
    ap.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--dwell", type=float, default=1.0, help="seconds inside zone before alerting")
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds before same ID can alert again")
    ap.add_argument("--events-dir", default="events", help="where snapshots and the CSV log go")
    ap.add_argument("--save-video", action="store_true", help="save annotated output video")
    ap.add_argument("--no-show", action="store_true", help="don't open a display window")
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU (auto if unset)")
    args = ap.parse_args()

    src, is_live = parse_source(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    zone = load_zone(args.zone, frame_w, frame_h)
    model = YOLO(args.model)
    try:
        class_ids = resolve_classes(model.names, args.classes)
    except ValueError as e:
        raise SystemExit(str(e))
    print("Watching classes:", "all" if class_ids is None else [model.names[i] for i in class_ids])

    monitor = ZoneMonitor(model, zone, class_ids, conf=args.conf, imgsz=args.imgsz,
                          tracker=args.tracker, dwell=args.dwell, cooldown=args.cooldown,
                          device=args.device)
    events_dir = Path(args.events_dir)
    log = EventLog(events_dir)

    writer = None
    if args.save_video:
        out_path = events_dir / f"output_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    frame_idx = 0
    start_wall = time.time()
    fps_t0, fps_frames, shown_fps = time.time(), 0, 0.0

    print(f"Running on {args.source} ({frame_w}x{frame_h}). Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live:
                    print("Stream read failed, retrying...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(src)
                    continue
                break
            frame_idx += 1
            # Video files: use video time so dwell/cooldown work the same at any processing speed
            now = (time.time() - start_wall) if is_live else frame_idx / fps

            res = monitor.update(frame, now)
            for alert in res.alerts:
                snap = log.write(alert)
                d = alert.det
                print(f"[ALERT] {alert.time:%H:%M:%S} intruder {d.cls} ID {d.track_id} (conf {d.conf:.2f}) -> {snap}")
                # TODO: send notification here (Telegram, siren, dashboard...)
            frame = res.frame

            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                shown_fps = fps_frames / (time.time() - fps_t0)
                fps_t0, fps_frames = time.time(), 0
            cv2.putText(frame, f"{shown_fps:.1f} FPS", (frame_w - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer:
                writer.write(frame)
            if not args.no_show:
                cv2.imshow("Intrusion detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        log.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
