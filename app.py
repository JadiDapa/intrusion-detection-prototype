"""
Streamlit web UI for the restricted-zone intrusion detector.

    streamlit run app.py

- Zone tab:      click on a video frame to draw / edit the polygon (defaults to zone.json)
- Detect tab:    run detection with a live preview, then explore stats, charts and alerts
- Sidebar:       pick the video (defaults to street.mp4) and tune model / rules
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

from draw_zone import is_self_intersecting
from intrusion import EventLog, ZoneMonitor, parse_source, resolve_classes, scale_zone

ROOT = Path(__file__).parent
DEFAULT_VIDEO = ROOT / "street.mp4"
DEFAULT_ZONE = ROOT / "zone.json"
UPLOAD_DIR = ROOT / "events" / "uploads"
RUNS_DIR = ROOT / "events" / "runs"
PREVIEW_W = 960      # live preview / heatmap width
EDITOR_W = 1280      # zone editor image width

# Chart colors (validated categorical slots + reserved "critical" status red)
C_FRAME = "#2a78d6"  # people in frame
C_ZONE = "#eb6834"   # people in zone
C_ALERT = "#e34948"  # intrusions / alerts
C_MUTED = "#9a9893"

st.set_page_config(page_title="Intrusion Detection", page_icon="🚨", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def probe(source: str):
    src, is_live = parse_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    info = dict(fps=fps, w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frames=0 if is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), is_live=is_live)
    info["duration"] = info["frames"] / fps if info["frames"] else 0.0
    cap.release()
    return info


@st.cache_data(show_spinner=False, ttl=60, max_entries=64)
def grab_frame(source: str, t: float):
    """RGB frame at t seconds (video files) or a fresh frame (live sources)."""
    src, is_live = parse_source(source)
    cap = cv2.VideoCapture(src)
    frame = None
    if is_live:
        for _ in range(10):  # webcams start dark
            ok, f = cap.read()
            frame = f if ok else frame
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        frame = frame if ok else None
    cap.release()
    return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Loading model...")
def model_names(model_path: str) -> dict:
    return YOLO(model_path).names


def resize_w(img, width):
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def to_browser_mp4(raw: Path, out: Path) -> bool:
    """OpenCV writes mp4v, which browsers can't play; re-encode to H.264."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-vf", "scale='min(1280,iw)':-2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def load_default_zone(w, h):
    if DEFAULT_ZONE.exists():
        data = json.loads(DEFAULT_ZONE.read_text())
        return scale_zone(data, w, h).tolist()
    return []


# ---------------------------------------------------------------- sidebar: source
st.sidebar.title("🚨 Intrusion detection")
st.sidebar.subheader("Video source")
src_mode = st.sidebar.radio("Source", ["Default (street.mp4)", "Upload video", "Path / RTSP URL / webcam"],
                            label_visibility="collapsed")
source = None
if src_mode == "Default (street.mp4)":
    source = str(DEFAULT_VIDEO)
elif src_mode == "Upload video":
    up = st.sidebar.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"])
    if up is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / Path(up.name).name
        if not dest.exists() or dest.stat().st_size != up.size:
            dest.write_bytes(up.getbuffer())
        source = str(dest)
else:
    typed = st.sidebar.text_input("Path, URL or webcam index", placeholder="0  or  rtsp://user:pass@ip/stream")
    source = typed.strip() or None

info = probe(source) if source else None
if source and info is None:
    st.sidebar.error(f"Could not open: {source}")
elif info:
    st.sidebar.caption(
        f"{info['w']}×{info['h']} · {info['fps']:.0f} fps"
        + ("" if info["is_live"] else f" · {info['duration']:.1f}s · {info['frames']} frames")
    )

# Zone state lives in the current video's pixel coordinates; rescale when the video changes.
ss = st.session_state
if info:
    if "zone" not in ss:
        ss.zone = dict(frame_width=info["w"], frame_height=info["h"], polygon=load_default_zone(info["w"], info["h"]))
    elif (ss.zone["frame_width"], ss.zone["frame_height"]) != (info["w"], info["h"]):
        pts = scale_zone(ss.zone, info["w"], info["h"]).tolist() if ss.zone["polygon"] else []
        ss.zone = dict(frame_width=info["w"], frame_height=info["h"], polygon=pts)
ss.setdefault("click_key", 0)
ss.setdefault("last_click", None)

# ---------------------------------------------------------------- sidebar: settings
st.sidebar.subheader("Model")
pt_files = sorted(p.name for p in ROOT.glob("*.pt"))
model_opts = pt_files + [m for m in ("yolo26n.pt", "yolo26s.pt") if m not in pt_files]
model_path = st.sidebar.selectbox("Weights", model_opts, index=model_opts.index("best.pt") if "best.pt" in model_opts else 0,
                                  help="Local .pt files, or COCO models (auto-download)")
names = model_names(model_path)
all_classes = list(names.values())
default_cls = ["person"] if "person" in all_classes else all_classes
classes = st.sidebar.multiselect("Classes that count as intruders", all_classes, default=default_cls)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
imgsz = st.sidebar.select_slider("Inference size", [320, 480, 640, 800, 960, 1280], value=640,
                                 help="Larger finds small/distant people but is slower")
tracker = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])

st.sidebar.subheader("Rules")
dwell = st.sidebar.number_input("Dwell before alert (s)", 0.0, 60.0, 1.0, 0.5)
cooldown = st.sidebar.number_input("Cooldown per person (s)", 0.0, 600.0, 30.0, 5.0)

st.sidebar.subheader("Performance")
stride = st.sidebar.number_input("Process every Nth frame", 1, 10, 2,
                                 help="2 = skip every other frame (≈2× faster). Timing still uses video time.")
save_video = st.sidebar.checkbox("Save annotated video", value=True)

if not info:
    st.info("Choose a video source in the sidebar to get started.")
    st.stop()

W, H = info["w"], info["h"]
poly = ss.zone["polygon"]
crossing = is_self_intersecting(poly)
zone_ok = len(poly) >= 3 and not crossing

tab_zone, tab_detect = st.tabs(["① Zone", "② Detect & dashboard"])

# ---------------------------------------------------------------- zone editor
with tab_zone:
    left, right = st.columns([3, 1], gap="large")
    with right:
        st.markdown("**Edit the restricted zone**")
        st.caption("Click on the image to add corners in order around the edge.")
        t = 0.0
        if not info["is_live"] and info["duration"] > 0:
            t = st.slider("Frame at (s)", 0.0, max(info["duration"] - 0.1, 0.0), min(5.0, info["duration"] / 2), 0.5)
        elif st.button("Grab new frame", width="stretch"):
            grab_frame.clear()

        def _reset_clicks():
            ss.click_key += 1
            ss.last_click = None

        c1, c2 = st.columns(2)
        if c1.button("↶ Undo", width="stretch", disabled=not poly):
            poly.pop()
            st.rerun()
        if c2.button("✕ Clear", width="stretch", disabled=not poly):
            poly.clear()
            _reset_clicks()
            st.rerun()
        c3, c4 = st.columns(2)
        if c3.button("Reset", width="stretch", help="Reload zone.json", disabled=not DEFAULT_ZONE.exists()):
            ss.zone["polygon"] = load_default_zone(W, H)
            _reset_clicks()
            st.rerun()
        if c4.button("💾 Save", width="stretch", type="primary", disabled=not zone_ok,
                     help="Overwrite zone.json (also used by detect.py)"):
            DEFAULT_ZONE.write_text(json.dumps(ss.zone, indent=2))
            st.toast("Saved zone.json")

        if crossing:
            st.warning("The edges cross each other. Undo and click the corners in order around the edge.")
        elif len(poly) < 3:
            st.warning(f"Need at least 3 points ({len(poly)} so far).")
        else:
            area = cv2.contourArea(np.array(poly, np.float32)) / (W * H) * 100
            st.success(f"{len(poly)} points · covers {area:.1f}% of the frame")

        st.download_button("Download zone JSON", json.dumps(ss.zone, indent=2), "zone.json",
                           "application/json", width="stretch", disabled=not poly)

        def _load_zone_upload():
            f = ss.get("zone_upload")
            if f is None:
                return
            try:
                data = json.loads(f.getvalue())
                ss.zone = dict(frame_width=W, frame_height=H, polygon=scale_zone(data, W, H).tolist())
                _reset_clicks()
            except Exception as e:
                ss.zone_upload_error = f"Invalid zone file: {e}"

        st.file_uploader("Load zone JSON", type=["json"], key="zone_upload", on_change=_load_zone_upload)
        if err := ss.pop("zone_upload_error", None):
            st.error(err)

        with st.expander("Points (pixels)"):
            st.dataframe(pd.DataFrame(poly, columns=["x", "y"], index=range(1, len(poly) + 1)) if poly
                         else pd.DataFrame(columns=["x", "y"]), height=220)

    with left:
        frame = grab_frame(source, t)
        if frame is None:
            st.error("Could not read a frame from this source.")
        else:
            img = resize_w(frame, EDITOR_W).copy()
            s = img.shape[1] / W
            pts = (np.array(poly, np.float32) * s).astype(np.int32) if poly else np.zeros((0, 2), np.int32)
            color = (255, 165, 0) if crossing else (255, 60, 60)
            if len(pts) >= 3:
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], len(pts) >= 3, color, 2)
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(int(v) for v in p), 6, (255, 255, 0), -1)
                cv2.putText(img, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(img, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            click = streamlit_image_coordinates(img, width="stretch", key=f"zone_click_{ss.click_key}",
                                                cursor="crosshair", image_format="JPEG", jpeg_quality=85)
            if click and click.get("unix_time") != ss.last_click:
                ss.last_click = click.get("unix_time")
                x = int(round(click["x"] * W / click["width"]))
                y = int(round(click["y"] * H / click["height"]))
                poly.append([min(max(x, 0), W - 1), min(max(y, 0), H - 1)])
                st.rerun()


# ---------------------------------------------------------------- charts
def axis_cfg(chart):
    return (chart.configure_axis(grid=True, gridColor="#8883", domain=False, tickColor="#8886",
                                 labelColor=C_MUTED, titleColor=C_MUTED, labelFontSize=11, titleFontSize=11)
            .configure_view(stroke=None)
            .configure_legend(orient="top", labelColor=C_MUTED, title=None, symbolStrokeWidth=3))


def occupancy_chart(tl: pd.DataFrame, height=260):
    """People in frame vs in zone over time, intrusion periods shaded."""
    long = tl.melt("t", ["in_frame", "in_zone"], var_name="series", value_name="people")
    long["series"] = long["series"].map({"in_frame": "In frame", "in_zone": "In zone"})
    color = alt.Color("series:N", scale=alt.Scale(domain=["In frame", "In zone"], range=[C_FRAME, C_ZONE]))
    x = alt.X("t:Q", title="Video time (s)")

    intr = tl[tl["intruders"] > 0][["t"]].copy()
    dt = float(tl["t"].diff().median()) if len(tl) > 1 else 0.1
    intr["t2"] = intr["t"] + dt
    shade = alt.Chart(intr).mark_rect(color=C_ALERT, opacity=0.13).encode(x="t:Q", x2="t2:Q")

    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    lines = alt.Chart(long).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=x, y=alt.Y("people:Q", title="People", axis=alt.Axis(tickMinStep=1)), color=color)
    points = lines.mark_point(size=60, filled=True, opacity=1).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = alt.Chart(tl).mark_rule(color=C_MUTED).encode(
        x="t:Q",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("t:Q", title="Time (s)", format=".1f"),
                 alt.Tooltip("in_frame:Q", title="In frame"),
                 alt.Tooltip("in_zone:Q", title="In zone"),
                 alt.Tooltip("intruders:Q", title="Intruders (confirmed)")],
    ).add_params(hover)
    return axis_cfg((shade + lines + points + rule).properties(height=height))


def alerts_chart(alerts: pd.DataFrame, duration: float):
    span = max(duration, alerts["source_time_s"].max() if len(alerts) else 1)
    step = next(s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1e9) if span / s <= 30)
    df = alerts.assign(bin=(alerts["source_time_s"] // step) * step)
    counts = df.groupby("bin").size().reindex(np.arange(0, span + step, step), fill_value=0)
    counts = counts.rename_axis("bin").reset_index(name="alerts")
    counts["label"] = counts["bin"].map(lambda b: f"{b:.0f}–{b + step:.0f}s")
    top = int(counts["alerts"].max())
    chart = alt.Chart(counts).mark_bar(color=C_ALERT, cornerRadiusEnd=4, width={"band": 0.85}).encode(
        x=alt.X("bin:O", title=f"Video time ({step:g}s bins)", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("alerts:Q", title="Alerts", scale=alt.Scale(domain=[0, max(top, 2)]),
                axis=alt.Axis(values=list(range(0, max(top, 2) + 1)) if top <= 10 else None, tickMinStep=1, format="d")),
        tooltip=[alt.Tooltip("label:N", title="Window"), alt.Tooltip("alerts:Q", title="Alerts")],
    )
    return axis_cfg(chart.properties(height=240))


def dwell_chart(tracks: pd.DataFrame, top=15):
    df = tracks[tracks["zone_time"] > 0].nlargest(top, "zone_time").copy()
    df["who"] = "ID " + df["track_id"].astype(str)
    df["status"] = np.where(df["alerts"] > 0, "Alerted", "Below dwell threshold")
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.75}).encode(
        y=alt.Y("who:N", sort="-x", title=None),
        x=alt.X("zone_time:Q", title="Seconds inside zone"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Alerted", "Below dwell threshold"],
                                                    range=[C_ALERT, C_MUTED])),
        tooltip=[alt.Tooltip("who:N", title="Track"), alt.Tooltip("zone_time:Q", title="Time in zone (s)", format=".1f"),
                 alt.Tooltip("max_dwell:Q", title="Longest stay (s)", format=".1f"),
                 alt.Tooltip("alerts:Q", title="Alerts"), alt.Tooltip("max_conf:Q", title="Max conf", format=".2f")],
    )
    return axis_cfg(chart.properties(height=max(120, 24 * len(df))))


def conf_chart(confs: list, threshold: float):
    df = pd.DataFrame({"conf": confs})
    bars = alt.Chart(df).mark_bar(color=C_FRAME, cornerRadiusEnd=3).encode(
        x=alt.X("conf:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="Detection confidence"),
        y=alt.Y("count():Q", title="Detections"),
        tooltip=[alt.Tooltip("conf:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="Confidence"),
                 alt.Tooltip("count():Q", title="Detections")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color=C_MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    return axis_cfg((bars + rule).properties(height=240))


def heatmap(run) -> np.ndarray:
    """Foot-position density over the frame (single-hue: brighter orange = more traffic)."""
    bg = run["background"].astype(np.float32)
    h, w = bg.shape[:2]
    s = w / run["w"]
    grid = np.zeros((h, w), np.float32)
    if run["feet"]:
        f = (np.array(run["feet"], np.float32) * s).astype(int)
        np.add.at(grid, (np.clip(f[:, 1], 0, h - 1), np.clip(f[:, 0], 0, w - 1)), 1)
        grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=w / 80)
        grid /= grid.max() or 1
    alpha = (np.sqrt(grid) * 0.85)[..., None]
    gray = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
    base = np.repeat(gray, 3, axis=2) * 0.55
    out = base * (1 - alpha) + np.array([235, 104, 52], np.float32) * alpha
    out = out.clip(0, 255).astype(np.uint8)
    cv2.polylines(out, [(np.array(run["zone"]) * s).astype(np.int32)], True, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------- stats
def tracks_df(run) -> pd.DataFrame:
    rows = [dict(track_id=k, **v) for k, v in run["tracks"].items()]
    cols = ["track_id", "cls", "first", "last", "zone_time", "max_dwell", "alerts", "max_conf"]
    return pd.DataFrame(rows, columns=cols)


def render_kpis(run):
    tl = pd.DataFrame(run["timeline"])
    tr = tracks_df(run)
    entered = tr[tr["zone_time"] > 0]
    elapsed = run["wall_elapsed"] or 1e-9
    spark = tl["in_zone"].tolist()[-200:] if len(tl) else None
    k = st.columns(6)
    k[0].metric("🚨 Alerts", len(run["alerts"]), border=True)
    k[1].metric("People tracked", len(tr), border=True, help="Unique tracker IDs (an ID can change if tracking is lost)")
    k[2].metric("Entered zone", len(entered), border=True)
    k[3].metric("Peak in zone", int(tl["in_zone"].max()) if len(tl) else 0, border=True,
                chart_data=spark, chart_type="area")
    k[4].metric("Avg time in zone", f"{entered['zone_time'].mean():.1f}s" if len(entered) else "–", border=True,
                help="Among people who entered the zone")
    k[5].metric("Processing speed", f"{run['processed'] / elapsed:.1f} fps", border=True,
                help=f"{run['processed']} frames analysed in {elapsed:.0f}s")


def render_dashboard(run):
    status = {"finished": "✅ Finished", "stopped": "⏹ Stopped early", "running": "⏳ Running",
              "error": "❌ Error"}[run["status"]]
    st.markdown(f"#### Results · {status}")
    s = run["settings"]
    st.caption(f"{run['source']} · model {s['model']} · conf {s['conf']} · dwell {s['dwell']}s · "
               f"cooldown {s['cooldown']}s · every {s['stride']} frame(s) · saved to `{run['dir'].relative_to(ROOT)}`")
    if run.get("error"):
        st.error(run["error"])
    render_kpis(run)

    tl = pd.DataFrame(run["timeline"])
    alerts = pd.DataFrame(run["alerts"], columns=["timestamp", "source_time_s", "track_id", "class", "confidence",
                                                  "x1", "y1", "x2", "y2", "snapshot"])
    tr = tracks_df(run)
    duration = float(tl["t"].max()) if len(tl) else 0.0

    if len(tl):
        st.markdown("**People over time** · red bands = confirmed intrusion")
        st.altair_chart(occupancy_chart(tl), width="stretch")
        intr_time = (tl["intruders"] > 0).mean() * duration
        busy = tl["in_frame"].mean()
        st.caption(f"Intrusion active for **{intr_time:.1f}s** of {duration:.1f}s "
                   f"({(tl['intruders'] > 0).mean() * 100:.0f}%) · avg {busy:.1f} people in frame, "
                   f"{tl['in_zone'].mean():.2f} in zone")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Alerts over time**")
        if len(alerts):
            st.altair_chart(alerts_chart(alerts, duration), width="stretch")
        else:
            st.info("No alerts in this run.")
    with b:
        st.markdown("**Time inside zone per person** · top 15")
        if (tr["zone_time"] > 0).any():
            st.altair_chart(dwell_chart(tr), width="stretch")
        else:
            st.info("Nobody entered the zone.")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Foot-traffic heatmap** · brighter = more time spent there")
        if run.get("background") is not None:
            st.image(heatmap(run), width="stretch")
    with b:
        st.markdown("**Detection confidence** · dashed line = threshold")
        if run["confs"]:
            st.altair_chart(conf_chart(run["confs"], s["conf"]), width="stretch")
            st.caption("Many detections just above the threshold? Raising it may cut false alarms.")

    st.markdown("**Alert log**")
    if len(alerts):
        st.dataframe(alerts.drop(columns=["snapshot"]), hide_index=True,
                     column_config={"source_time_s": st.column_config.NumberColumn("Video time (s)", format="%.1f"),
                                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1,
                                                                                  format="%.2f")})
        with st.expander(f"Snapshots ({len(alerts)})", expanded=True):
            cols = st.columns(4)
            for i, r in enumerate(alerts.itertuples()):
                if Path(r.snapshot).exists():
                    cols[i % 4].image(resize_w(cv2.cvtColor(cv2.imread(r.snapshot), cv2.COLOR_BGR2RGB), 480),
                                      caption=f"ID {r.track_id} · {r.source_time_s:.1f}s · conf {float(r.confidence):.2f}")
    else:
        st.caption("No alerts.")

    with st.expander("Per-person table"):
        st.dataframe(tr.round(2), hide_index=True)

    st.markdown("**Downloads**")
    d = st.columns(4)
    ev = run["dir"] / "events.csv"
    if ev.exists():
        d[0].download_button("Alert log (CSV)", ev.read_bytes(), "events.csv", "text/csv", width="stretch")
    d[1].download_button("Timeline (CSV)", tl.to_csv(index=False), "timeline.csv", "text/csv", width="stretch")
    d[2].download_button("Per-person (CSV)", tr.to_csv(index=False), "people.csv", "text/csv", width="stretch")

    if run.get("video_raw") and run["status"] != "running":
        mp4 = run["dir"] / "output.mp4"
        if not mp4.exists():
            with st.spinner("Encoding annotated video for the browser..."):
                if not to_browser_mp4(run["video_raw"], mp4):
                    mp4 = None
        if mp4 and mp4.exists():
            d[3].download_button("Annotated video", mp4.read_bytes(), "output.mp4", "video/mp4", width="stretch")
            st.markdown("**Annotated video**")
            st.video(str(mp4))
        else:
            st.warning(f"Couldn't re-encode the video for playback. Raw file: {run['video_raw']}")


# ---------------------------------------------------------------- detection run
def run_detection(live_img, live_stats, progress, background):
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run = dict(
        dir=run_dir, source=Path(source).name if not info["is_live"] else source, w=W, h=H, fps=info["fps"],
        settings=dict(model=model_path, conf=conf, imgsz=imgsz, dwell=dwell, cooldown=cooldown, stride=stride),
        status="running", error=None, zone=[list(p) for p in poly],
        timeline=[], tracks={}, alerts=[], feet=[], confs=[], processed=0,
        wall_start=time.time(), wall_elapsed=0.0, video_raw=None,
        background=None if background is None else resize_w(background, PREVIEW_W),
    )
    ss.run = run
    log = EventLog(run_dir)
    model = YOLO(model_path)  # fresh model = fresh tracker state
    class_ids = resolve_classes(names, ",".join(classes)) if classes else None
    monitor = ZoneMonitor(model, np.array(poly, np.int32), class_ids, conf=conf, imgsz=imgsz, tracker=tracker,
                          dwell=dwell, cooldown=cooldown)
    src, is_live = parse_source(source)
    cap = cv2.VideoCapture(src)
    fps, total = info["fps"], info["frames"]
    writer = None
    if save_video:
        run["video_raw"] = run_dir / "output_raw.mp4"
        writer = cv2.VideoWriter(str(run["video_raw"]), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))

    frame_idx, prev_now, last_ui = 0, None, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live:
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(src)
                    continue
                break
            frame_idx += 1
            if (frame_idx - 1) % stride:
                continue
            now = (time.time() - run["wall_start"]) if is_live else frame_idx / fps
            dt = 0.0 if prev_now is None else now - prev_now
            prev_now = now
            if run["background"] is None:
                run["background"] = resize_w(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), PREVIEW_W)

            res = monitor.update(frame, now)
            for alert in res.alerts:
                snap = log.write(alert)
                d = alert.det
                run["alerts"].append([alert.time.isoformat(timespec="seconds"), round(now, 2), d.track_id, d.cls,
                                      round(d.conf, 2), *d.box, str(snap)])
                st.toast(f"🚨 Intruder ID {d.track_id} at {now:.1f}s")
            for d in res.detections:
                tr = run["tracks"].setdefault(d.track_id, dict(cls=d.cls, first=now, last=now, zone_time=0.0,
                                                               max_dwell=0.0, alerts=0, max_conf=0.0))
                tr["last"] = now
                tr["max_conf"] = max(tr["max_conf"], d.conf)
                if d.inside:
                    tr["zone_time"] += dt
                    tr["max_dwell"] = max(tr["max_dwell"], d.dwell)
                run["feet"].append(d.foot)
                run["confs"].append(round(d.conf, 3))
            for alert in res.alerts:
                run["tracks"][alert.det.track_id]["alerts"] += 1
            run["timeline"].append(dict(t=round(now, 2), in_frame=len(res.detections), in_zone=res.in_zone,
                                        intruders=res.intruders))
            run["processed"] += 1
            run["wall_elapsed"] = time.time() - run["wall_start"]

            if writer:
                writer.write(res.frame)
            live_img.image(cv2.cvtColor(resize_w(res.frame, PREVIEW_W), cv2.COLOR_BGR2RGB), width="stretch")
            if total:
                progress.progress(min(frame_idx / total, 1.0), text=f"{now:.1f}s / {info['duration']:.1f}s")
            if time.time() - last_ui > 1.5:
                last_ui = time.time()
                with live_stats.container():
                    render_kpis(run)
                    st.altair_chart(occupancy_chart(pd.DataFrame(run["timeline"]), height=200), width="stretch")
        run["status"] = "finished"
    except Exception as e:
        run["status"], run["error"] = "error", f"{type(e).__name__}: {e}"
    finally:
        # Also runs when Streamlit interrupts the script (Stop button / any widget change)
        if run["status"] == "running":
            run["status"] = "stopped"
        run["wall_elapsed"] = time.time() - run["wall_start"]
        cap.release()
        if writer:
            writer.release()
        log.close()


with tab_detect:
    run = ss.get("run")
    c1, c2, c3 = st.columns([1, 1, 4])
    start = c1.button("▶ Start detection", type="primary", width="stretch", disabled=not zone_ok or not classes)
    c2.button("⏹ Stop", width="stretch", help="Stops the run; results so far are kept")
    if not zone_ok:
        c3.warning("Draw a valid zone (3+ points, no crossing edges) in the Zone tab first.")
    elif not classes:
        c3.warning("Pick at least one class in the sidebar.")
    else:
        c3.caption(f"Model **{model_path}** on **{Path(source).name if not info['is_live'] else source}** · "
                   f"{len(poly)}-point zone. Changing any setting while running stops the run.")

    if start:
        progress = st.empty() if info["is_live"] else st.progress(0.0, text="Starting...")
        v, s_ = st.columns([3, 2], gap="large")
        live_img = v.empty()
        live_stats = s_.empty()
        # Heatmap backdrop: the frame shown in the zone editor (live sources use their first frame)
        run_detection(live_img, live_stats, progress, None if info["is_live"] else grab_frame(source, t))
        st.rerun()  # redraw the page with the full dashboard
    elif run:
        render_dashboard(run)
    else:
        st.info("Press **Start detection** to analyse the video. A live preview and stats appear here while it runs.")
