"""
Draw the restricted zone polygon on a frame from your camera/video.

Usage:
    python draw_zone.py --source street.mp4              # frame at 2 s (default)
    python draw_zone.py --source street.mp4 --time 10    # frame at 10 s
    python draw_zone.py --source 0                       # webcam
    python draw_zone.py --source rtsp://...              # IP camera

Controls:
    Left click   add a point (click the corners IN ORDER around the edge)
    Right click  undo last point (or press 'u')
    n / p        next / previous frame (+/- 1 second, video files only)
    c            clear all points
    s            save zone to JSON and quit
    q / Esc      quit without saving
"""

import argparse
import json

import cv2
import numpy as np


def open_source(source: str):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    is_file = not isinstance(src, int) and not str(src).lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )
    return cap, is_file


def read_at(cap, is_file: bool, t: float):
    """Read a frame at t seconds (video files) or a fresh frame (live streams)."""
    if is_file:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    frame = None
    for _ in range(10):  # live: skip a few frames, webcams start dark
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


def video_duration(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    return n / fps if fps > 0 else 0.0


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return (orient(p1, p2, p3) != orient(p1, p2, p4)) and (
        orient(p3, p4, p1) != orient(p3, p4, p2)
    )


def is_self_intersecting(points) -> bool:
    """True if the polygon's edges cross each other (e.g. a bow-tie shape)."""
    n = len(points)
    if n < 4:
        return False
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbouring edges share a corner
            if _segments_cross(*edges[i], *edges[j]):
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Draw a restricted zone polygon")
    ap.add_argument(
        "--source", default="0", help="webcam index, video path, or RTSP URL"
    )
    ap.add_argument(
        "--time",
        type=float,
        default=5.0,
        help="seconds into the video to grab the frame",
    )
    ap.add_argument("--out", default="zone.json", help="where to save the polygon")
    args = ap.parse_args()

    cap, is_file = open_source(args.source)
    duration = video_duration(cap) if is_file else 0.0
    t = min(args.time, max(duration - 0.1, 0)) if duration else args.time

    frame = read_at(cap, is_file, t)
    if frame is None:
        raise SystemExit(f"Could not read a frame at {t:.1f}s from {args.source}")
    h, w = frame.shape[:2]
    points: list[list[int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = "Draw zone  |  L-click: add  R-click/u: undo  n/p: next/prev sec  c: clear  s: save  q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), int(min(w, 1280) * h / w))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        crossing = is_self_intersecting(points)
        color = (0, 165, 255) if crossing else (0, 0, 255)
        if len(points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.array(points, np.int32)], color)
            canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
        if len(points) >= 2:
            cv2.polylines(
                canvas, [np.array(points, np.int32)], len(points) >= 3, color, 2
            )
        for i, p in enumerate(points):
            cv2.circle(canvas, tuple(p), 6, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        info = f"points: {len(points)}"
        if is_file:
            info += f"   time: {t:.1f}s" + (f" / {duration:.1f}s" if duration else "")
        cv2.putText(canvas, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(
            canvas, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        if crossing:
            msg = "Edges cross! Click corners in order around the edge (u to undo)"
            cv2.putText(
                canvas, msg, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4
            )
            cv2.putText(
                canvas, msg, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
            )
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if key == ord("u") and points:
            points.pop()
        if key == ord("c"):
            points.clear()
        if key in (ord("n"), ord("p")):
            if is_file:
                new_t = t + (1.0 if key == ord("n") else -1.0)
                if duration:
                    new_t = min(new_t, max(duration - 0.1, 0))
                f = read_at(cap, is_file, max(new_t, 0))
                if f is not None:
                    frame, t = f, max(new_t, 0)
            else:
                f = read_at(cap, is_file, 0)
                if f is not None:
                    frame = f
        if key == ord("s"):
            if len(points) < 3:
                print("Need at least 3 points for a polygon.")
                continue
            if crossing:
                print(
                    "The zone's edges cross each other. Undo and click the corners in order around the edge."
                )
                continue
            data = {"frame_width": w, "frame_height": h, "polygon": points}
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved zone with {len(points)} points to {args.out}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
