"""
Camera acquisition, ISP optimization, and image quality adjustment engine for Jetson Nano.
Includes fixed hardware/software color calibration:
- Brightness: 8, Contrast: 1.32, Saturation: 0.52, Hue: +6, Sharpness: 0.98, IR Filter: 0%
- GStreamer Hardware ISP pipeline with flip-method=2 (sensor upright un-mirrored)
- Multi-WB modes with dynamic switching
"""

import os
import cv2
import numpy as np
import time

# ---------------------------------------------------------
# WB Modes
# ---------------------------------------------------------
wb_modes = [
    (0, "Auto"),
    (1, "Incandescent"),
    (2, "Fluorescent"),
    (3, "Warm Fluorescent"),
    (4, "Daylight"),
    (5, "Cloudy"),
    (6, "Twilight"),
    (7, "Shade"),
]
current_wb = 0

# ---------------------------------------------------------
# FIXED CAMERA SETTINGS
# ---------------------------------------------------------
brightness = 3          # Slider value 58 (beta)
contrast = 1.32         # Slider value 66 (alpha)
saturation = 0.52       # Slider value 26
hue = 6                 # Slider value 56
sharpness = 0.98        # Slider value 49
ir_strength = 0.0       # Slider value 0 (IR Filter: 0%)

# ---------------------------------------------------------
# Resolutions & Framerates
# ---------------------------------------------------------
RESOLUTIONS = [
    (1920, 1080, "1080p Full HD"),
    (1280, 720,  "720p HD"),
    (640,  480,  "480p SD"),
]
current_res_idx = 0
FRAMERATES = [30, 60, 15]
current_fps_idx = 0


# ---------------------------------------------------------
# Frame Adjustments (Brightness, Contrast, IR, Saturation, Hue, Sharpness)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Frame Adjustments (Brightness, Contrast, IR, Saturation, Hue, Sharpness)
# ---------------------------------------------------------
_lut_cache = {}


def _get_luts(sat, hue):
    key = (round(float(sat), 3), int(round(float(hue))))
    if key in _lut_cache:
        return _lut_cache[key]
    s_val, h_val = key
    sat_lut = np.clip(np.arange(256, dtype=np.float32) * s_val, 0, 255).astype(np.uint8)
    hue_lut = np.clip((np.arange(256, dtype=np.int32) + h_val) % 180, 0, 179).astype(np.uint8)
    _lut_cache[key] = (sat_lut, hue_lut)
    return sat_lut, hue_lut


def apply_frame_adjustments(
    frame: np.ndarray,
    c_contrast=None,
    c_brightness=None,
    c_ir=None,
    c_sat=None,
    c_hue=None,
    c_sharp=None,
    **kwargs,
) -> np.ndarray:
    """
    Applies the custom color calibration and image quality adjustment pipeline:
    1. Brightness & Contrast (convertScaleAbs)
    2. Software IR Correction
    3. Saturation & Hue in HSV space (accelerated with uint8 LUTs)
    4. Unsharp masking if sharpness > 1.0
    """
    if frame is None or frame.size == 0:
        return frame

    _contrast = contrast if c_contrast is None else c_contrast
    _brightness = brightness if c_brightness is None else c_brightness
    _ir = ir_strength if c_ir is None else c_ir
    _sat = saturation if c_sat is None else c_sat
    _hue = hue if c_hue is None else c_hue
    _sharp = sharpness if c_sharp is None else c_sharp

    # Fast path: neutral pass-through
    if (
        abs(_contrast - 1.0) < 0.01
        and abs(_brightness) < 0.01
        and _ir <= 0.001
        and abs(_sat - 1.0) < 0.01
        and abs(_hue) < 0.01
        and _sharp <= 1.001
    ):
        return frame

    frame_adj = frame

    # 1. Brightness and Contrast
    if abs(_contrast - 1.0) >= 0.01 or abs(_brightness) >= 0.01:
        frame_adj = cv2.convertScaleAbs(
            frame_adj,
            alpha=_contrast,
            beta=_brightness,
        )

    # 2. Software IR Correction
    if _ir > 0.001:
        b, g, r = cv2.split(frame_adj)
        scale = 1.0 - (0.5 * _ir)
        r_lut = np.clip(np.arange(256, dtype=np.float32) * scale, 0, 255).astype(np.uint8)
        corrected_r = cv2.LUT(r, r_lut)
        frame_adj = cv2.merge([b, g, corrected_r])

    # 3. Saturation and Hue
    if abs(_sat - 1.0) >= 0.01 or abs(_hue) >= 0.01:
        hsv = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2HSV)
        sat_lut, hue_lut = _get_luts(_sat, _hue)
        if abs(_sat - 1.0) >= 0.01:
            hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], sat_lut)
        if abs(_hue) >= 0.01:
            hsv[:, :, 0] = cv2.LUT(hsv[:, :, 0], hue_lut)
        frame_adj = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Sharpness Filter (if sharpness > 1)
    if _sharp > 1.001:
        amount = _sharp - 1.0
        kernel = np.array([
            [0, -1, 0],
            [-1, 5 + amount, -1],
            [0, -1, 0],
        ], dtype=np.float32)
        frame_adj = cv2.filter2D(frame_adj, -1, kernel)

    return frame_adj


# ---------------------------------------------------------
# GStreamer Pipeline (Optimized for IMX477 / IMX219 / CSI)
# ---------------------------------------------------------
def gstreamer_pipeline(
    wbmode=0,
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    framerate=60,
    flip_method=2,
):
    """
    Hardware ISP pipeline for Jetson Nano CSI cameras:
    tnr-mode=1 tnr-strength=0.30 ee-mode=1 ee-strength=0.75 aeantibanding=2 saturation=1.05
    with high-quality bicubic interpolation (interpolation-method=5) and max-buffers=2 for zero lag.
    """
    sensor_w = max(1920, capture_width)
    sensor_h = max(1080, capture_height)
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} wbmode={wbmode} tnr-mode=1 tnr-strength=0.30 ee-mode=1 ee-strength=0.75 aeantibanding=2 saturation=1.05 ! "
        f"video/x-raw(memory:NVMM), width=(int){sensor_w}, height=(int){sensor_h}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv interpolation-method=5 flip-method={flip_method} ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, format=(string)BGRx, framerate=(fraction){framerate}/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true sync=false max-buffers=2"
    )


def gstreamer_pipeline_csi(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    framerate=60,
    flip_method=2,
    wbmode=0,
):
    """Alias for CSI camera pipeline compatible with existing modules."""
    return gstreamer_pipeline(
        sensor_id=sensor_id,
        capture_width=capture_width,
        capture_height=capture_height,
        framerate=framerate,
        flip_method=flip_method,
        wbmode=wbmode,
    )


def gstreamer_pipeline_v4l2(sensor_id=0, width=1920, height=1080, framerate=60):
    """GStreamer pipeline for V4L2 USB cameras."""
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true sync=false max-buffers=2"
    )


# ---------------------------------------------------------
# Camera Hardware Discovery
# ---------------------------------------------------------
def list_available_cameras():
    """
    Enumerates available physical video capture devices on the Linux system.
    Returns a list of dicts: [{"id": int, "device": str, "name": str, "type": str}]
    """
    cams = []
    v4l_path = "/sys/class/video4linux"
    if os.path.exists(v4l_path):
        for entry in sorted(os.listdir(v4l_path)):
            if entry.startswith("video"):
                idx_str = entry.replace("video", "")
                if idx_str.isdigit():
                    dev_id = int(idx_str)
                    dev_node = f"/dev/{entry}"
                    name_file = os.path.join(v4l_path, entry, "name")
                    dev_name = entry
                    if os.path.exists(name_file):
                        try:
                            with open(name_file, "r") as f:
                                dev_name = f.read().strip()
                        except Exception:
                            pass
                    cams.append({
                        "id": dev_id,
                        "device": dev_node,
                        "name": dev_name,
                        "type": "v4l2",
                    })

    if not cams and os.path.exists("/dev"):
        for fname in sorted(os.listdir("/dev")):
            if fname.startswith("video") and fname[5:].isdigit():
                dev_id = int(fname[5:])
                dev_node = f"/dev/{fname}"
                cams.append({
                    "id": dev_id,
                    "device": dev_node,
                    "name": f"V4L2 Device {fname}",
                    "type": "v4l2",
                })
    return cams


# ---------------------------------------------------------
# Open Camera
# ---------------------------------------------------------
def open_camera(
    sensor_id=0,
    display_width=1920,
    display_height=1080,
    framerate=60,
    wb_idx=None,
    flip_method=2,
):
    """
    Opens CSI camera with GStreamer pipeline and configured WB mode and framerate,
    falling back to V4L2 if unavailable.
    Explicit simulated video files and RTSP streams are supported as distinct modes.
    """
    global current_wb
    if wb_idx is not None:
        current_wb = wb_idx % len(wb_modes)

    wb_val = wb_modes[current_wb][0]
    print(
        f"Opening camera (sensor_id={sensor_id}, {display_width}x{display_height} @ {framerate}fps, WB={wb_modes[current_wb][1]})..."
    )

    # 0. Check for RTSP stream, HTTP stream, or local video file loop
    # 0. Check for explicit RTSP / HTTP network stream
    if isinstance(sensor_id, str) and sensor_id.startswith(("rtsp://", "http://", "https://")):
        print(f"Opening network camera stream source: {sensor_id}...")
        cap = cv2.VideoCapture(sensor_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("Network camera stream opened successfully.")
                return cap, "rtsp_stream"
            cap.release()
        raise RuntimeError(f"Could not connect to network camera stream: {sensor_id}")

    # 1. Check for explicit local simulated video file
    if isinstance(sensor_id, str) and (not sensor_id.isdigit()):
        if sensor_id.startswith(("rtsp://", "http://", "https://")) or os.path.exists(sensor_id):
            print(f"Opening video/network camera source: {sensor_id}...")
        if os.path.exists(sensor_id):
            if os.path.getsize(sensor_id) < 1024:
                raise RuntimeError(f"Simulated video file is empty or corrupted: {sensor_id}")
            print(f"Opening explicit simulated video source: {sensor_id}...")
            cap = cv2.VideoCapture(sensor_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print("Video stream source opened successfully.")
                    return cap, "video_loop"
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print(f"Simulated video feed opened successfully ({os.path.basename(sensor_id)}).")
                    return cap, "simulated_video"
                cap.release()
            raise RuntimeError(f"Could not decode frames from simulated video file: {sensor_id}")
        else:
            raise RuntimeError(f"Configured camera file source not found: {sensor_id}")

    # 1. Try Jetson CSI camera with configured wbmode, hardware ISP, and flip-method
    # 2. Physical Hardware Camera Detection & Probing
    sensor_int = int(sensor_id) if (isinstance(sensor_id, int) or (isinstance(sensor_id, str) and sensor_id.isdigit())) else 0
    available_devices = list_available_cameras()

    # Step 2a: Try Jetson CSI camera with hardware ISP (nvarguscamerasrc)
    pipeline_csi = gstreamer_pipeline_csi(
        sensor_id=sensor_int,
        capture_width=display_width,
        capture_height=display_height,
        framerate=framerate,
        flip_method=flip_method,
        wbmode=wb_val,
    )
    cap = cv2.VideoCapture(pipeline_csi, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print("CSI Camera started successfully via GStreamer.")
            print(f"CSI Camera started successfully via GStreamer (sensor-id={sensor_int}).")
            return cap, "bgr"
        cap.release()
        time.sleep(0.1)

    # 2. Try USB Camera via GStreamer
    # Step 2b: Try USB Camera via GStreamer V4L2
    pipeline_v4l2 = gstreamer_pipeline_v4l2(
        sensor_int, display_width, display_height, framerate
    )
    cap = cv2.VideoCapture(pipeline_v4l2, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print("USB Camera started successfully via GStreamer V4L2 backend.")
            print(f"USB Camera started successfully via GStreamer V4L2 backend (/dev/video{sensor_int}).")
            return cap, "bgr"
        cap.release()
        time.sleep(0.1)

    # 3. Try native OpenCV V4L2 backend
    print("GStreamer failed. Trying native V4L2 backend...")
    # Step 2c: Try native OpenCV V4L2 backend
    print(f"Trying native V4L2 backend for /dev/video{sensor_int}...")
    cap = cv2.VideoCapture(sensor_int, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        cap.set(cv2.CAP_PROP_FPS, framerate)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print("USB Camera started successfully via V4L2 native backend.")
            print(f"USB Camera started successfully via V4L2 native backend (/dev/video{sensor_int}).")
            return cap, "v4l2_raw"
        cap.release()

    # Hardware camera initialization failed. Explicitly raise RuntimeError without silent video fallback.
    # Step 2d: If configured sensor_int failed, check if alternative cameras were detected
    if available_devices:
        for alt_cam in available_devices:
            alt_id = alt_cam["id"]
            if alt_id != sensor_int:
                print(f"Trying alternative detected camera: {alt_cam['device']} ({alt_cam['name']})...")
                alt_cap = cv2.VideoCapture(alt_id, cv2.CAP_V4L2)
                if alt_cap.isOpened():
                    alt_cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
                    alt_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
                    alt_cap.set(cv2.CAP_PROP_FPS, framerate)
                    ret, frame = alt_cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Connected to alternative camera {alt_cam['device']} ({alt_cam['name']}).")
                        return alt_cap, "v4l2_raw"
                    alt_cap.release()

    # No physical camera could be opened. Report detailed diagnostics.
    dev_list_str = ", ".join([f"{c['device']} ({c['name']})" for c in available_devices]) if available_devices else "None"
    raise RuntimeError(
        f"No camera sensor detected at `/dev/video{sensor_int}` (available physical devices: {dev_list_str}). "
        "Please check: 1) CSI ribbon cable is firmly seated with contacts facing the carrier board heatsink, "
        "2) Check the 'SUNNY' flex connector on the camera module PCB, "
        "3) Run 'sudo systemctl restart nvargus-daemon', 4) Plug in a USB webcam, or "
        "5) Set CAMERA_SENSOR_ID=dataset/VID_20260826_111523.mp4 in .env for simulated live streaming."
    )


# ---------------------------------------------------------
# Frame Reader with Adjustments
# ---------------------------------------------------------
def read_bgr_frame(cap, mode, height, width, apply_tuning=True):
    """
    Reads a BGR frame from capture device, applies color conversions, calibration adjustment,
    and resizing.
    """
    ret, frame = cap.read()
    if (not ret or frame is None) and mode in ("video_loop", "simulated_video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    if not ret or frame is None:
        return False, None

    if mode == "v4l2_raw":
        if len(frame.shape) == 2 or frame.shape[2] != 3:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
            except Exception:
                pass
        elif frame.shape[2] == 3:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            except Exception:
                pass

    if apply_tuning:
        frame = apply_frame_adjustments(frame)

    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))

    return True, frame


def create_display_window(window_name, width=800, height=450):
    """Helper for OpenCV GUI window creation."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)


# ---------------------------------------------------------
# Standalone Interactive Camera Quality & Adjustment Runner
# ---------------------------------------------------------
if __name__ == "__main__":
    cap, mode_str = open_camera(
        display_width=RESOLUTIONS[current_res_idx][0],
        display_height=RESOLUTIONS[current_res_idx][1],
        framerate=FRAMERATES[current_fps_idx],
    )

    if not cap.isOpened():
        print("Camera Failed")
        exit(1)

    print("=" * 65)
    print("Jetson Nano Camera Quality & Adjustment System")
    print(f"Fixed Settings -> Contrast: {contrast}, Brightness: {brightness}, Saturation: {saturation}, Hue: {hue}, Sharpness: {sharpness}, IR Filter: {int(ir_strength*100)}%")
    print("Controls:")
    print("  'W'       : Cycle White Balance Mode")
    print("  'R'       : Cycle Resolution (1080p -> 720p -> 480p)")
    print("  'Q' / ESC : Exit")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame Failed")
            break

        # Apply image quality adjustments
        frame_adj = apply_frame_adjustments(frame)

        # Display Information
        cv2.putText(
            frame_adj,
            f"WB: {wb_modes[current_wb][1]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame_adj,
            f"IR Filter: {int(ir_strength * 100)}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Camera", frame_adj)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('w') or key == ord('W'):
            current_wb = (current_wb + 1) % len(wb_modes)
            print(f"Switching WB Mode: {wb_modes[current_wb][1]}")
            cap.release()
            try:
                cap, mode_str = open_camera(
                    display_width=RESOLUTIONS[current_res_idx][0],
                    display_height=RESOLUTIONS[current_res_idx][1],
                    framerate=FRAMERATES[current_fps_idx],
                )
            except Exception as e:
                print(f"Camera re-open failed: {e}")
                break
        elif key == ord('r') or key == ord('R'):
            current_res_idx = (current_res_idx + 1) % len(RESOLUTIONS)
            print(f"Switching Resolution -> {RESOLUTIONS[current_res_idx][2]}")
            cap.release()
            try:
                cap, mode_str = open_camera(
                    display_width=RESOLUTIONS[current_res_idx][0],
                    display_height=RESOLUTIONS[current_res_idx][1],
                    framerate=FRAMERATES[current_fps_idx],
                )
            except Exception as e:
                print(f"Camera re-open failed: {e}")
                break

    cap.release()
    cv2.destroyAllWindows()
