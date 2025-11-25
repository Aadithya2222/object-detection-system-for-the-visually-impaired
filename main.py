import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time
import winsound
import threading
import os
import math

# Define class sizes early
class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}



def periodic_beep():
    while True:
        winsound.Beep(1000, 500)  # 1000Hz for 500ms
        time.sleep(10)  # Wait for 10 seconds

def speak(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 235)
    engine.setProperty('volume', 1.0)

    while True:
        if not q.empty():
            label, distance, position = q.get()
            rounded_distance = round(distance * 2) / 2  # Round to integer or in steps of 0.5
            
            # Special alert for vehicles or humans within 1 meter
            if distance <= 1:
                engine.setProperty('volume', 1.0)  # Maximum volume for alert
                if label in ["car", "bus", "motorcycle"]:
                    # Vehicle alert with higher pitch (2000Hz)
                    engine.say(f"WARNING! {label} VERY CLOSE! {rounded_distance} METERS {position}")
                    winsound.Beep(2000, 1000)  # High-pitched warning beep for vehicles
                elif label == "person":
                    # Person alert with different pitch (1500Hz)
                    engine.say(f"CAUTION! PERSON VERY CLOSE! {rounded_distance} METERS {position}")
                    winsound.Beep(1500, 800)  # Slightly different tone for humans
            # IF IT SAYS A INT NUMBER, IT REMOVES THE .0 PART. IT SAYS DIRECTLY 2 INSTEAD OF 2.0.
            rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
            if label in class_avg_sizes:
                engine.say(f"{label} IS {rounded_distance_str} METERS ON {position}")
                engine.runAndWait()
            with queue.mutex:
                queue.queue.clear()
        else:
            time.sleep(0.1)  # To avoid busy waiting
            
queue = Queue()
t = Thread(target=speak, args=(queue,))
beep_thread = Thread(target=periodic_beep, daemon=True)  # daemon=True ensures thread stops when main program stops
t.start()
beep_thread.start()

# old calculate_distance removed; use calculate_distance_from_pixels instead


def get_position(frame_width, box):
    if box[0] < frame_width // 3:
        return "LEFT"
    elif box[0] < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"


def blur_person(image, coords):
    # coords: [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(v) for v in coords]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    top_h = int(0.08 * h)
    if top_h <= 0:
        return image
    # guard bounds
    y_end = min(image.shape[0], y1 + top_h)
    x_end = min(image.shape[1], x1 + w)
    if y1 < 0 or x1 < 0 or y1 >= image.shape[0] or x1 >= image.shape[1]:
        return image
    top_region = image[y1:y_end, x1:x_end]
    if top_region.size == 0:
        return image
    blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
    image[y1:y_end, x1:x_end] = blurred_top_region
    return image


model = YOLO("gpModel.pt")
# Use IP camera stream - try multiple URL formats
stream_urls = [
    "http://10.186.111.110:8080/video",
    "http://10.186.111.110:8080/",
    "http://10.186.111.110:8080",
    "http://10.186.111.110/video",
]

cap = None
for url in stream_urls:
    print(f"Attempting to connect to: {url}")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        stream_url = url
        print(f"Successfully connected to: {url}")
        break
    cap.release()

# Check if the stream is opened successfully
if cap is None or not cap.isOpened():
    print("Error: Could not open IP camera stream. Trying with MJPEG stream format...")
    # Try MJPEG format
    stream_url = "http://10.186.111.110:8080/video?type=mjpeg"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open IP camera stream at {stream_url}")
        print("Please verify:")
        print("1. IP address is correct (currently: 10.186.111.110)")
        print("2. Port is correct (currently: 8080)")
        print("3. Camera app is running on the device")
        print("4. Device is connected to the same network")
        exit()

# Add a small delay to establish connection
time.sleep(2)

pause = False
# Force a fixed display resolution to avoid visual zooming when stream frames change size
TARGET_W, TARGET_H = 1280, 720

# Detection settings: run inference on a smaller image and scale boxes back to display size
DET_W, DET_H = 640, 360

# Asynchronous detection buffers and locks
frame_for_detection = None
new_frame_for_detection = False
frame_lock = threading.Lock()
latest_detections = None
results_lock = threading.Lock()
INFER_FPS = 5.0  # target inference rate (frames per second)

def detector_thread():
    global frame_for_detection, new_frame_for_detection, latest_detections
    infer_interval = 1.0 / INFER_FPS
    while True:
        start = time.time()
        # get the latest frame for detection
        with frame_lock:
            if frame_for_detection is None:
                # nothing to do
                pass
            else:
                frame_small = frame_for_detection.copy()
                new_frame_for_detection = False
        try:
            # if we have a frame to process
            if 'frame_small' in locals():
                # run model on smaller frame
                results = model.predict(frame_small)
                r = results[0]
                dets = []
                # scale factors from detection size to display size
                sx = TARGET_W / DET_W
                sy = TARGET_H / DET_H
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    label = r.names[cls_id]
                    xy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xy]
                    # scale coordinates
                    x1 = int(x1 * sx)
                    y1 = int(y1 * sy)
                    x2 = int(x2 * sx)
                    y2 = int(y2 * sy)
                    conf = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0
                    dets.append({"label": label, "xyxy": [x1, y1, x2, y2], "conf": conf})
                with results_lock:
                    latest_detections = dets
                # cleanup local
                del frame_small
        except Exception:
            pass
        elapsed = time.time() - start
        to_sleep = infer_interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

# start detector thread
det_thread = Thread(target=detector_thread, daemon=True)
det_thread.start()

def calculate_distance_from_pixels(pixel_width, frame_width, label):
    # similar formula as before but using provided pixel width
    object_width = pixel_width
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    focal_pixels = (frame_width * 0.5) / np.tan(np.radians(70 / 2))
    distance = focal_pixels / (object_width + 1e-6)
    return round(distance, 2)

# Simple stabilization state
prev_gray = None
prev_pts = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Recording settings: save 1-minute clips and overwrite the same file each cycle
# Use a temporary file and atomic replace so the output file is always playable
OUT_FILENAME = "latest_clip.avi"
TMP_FILENAME = "latest_clip.tmp.avi"
CLIP_DURATION = 60.0  # seconds
video_writer = None
clip_start_time = None
writer_tmp_path = TMP_FILENAME

# Frame rate measurement (to sync video playback speed)
frame_times = []  # Track last 30 frame capture times
measured_fps = 15.0  # Default fallback
last_frame_time = time.time()
frame_count = 0
# Track last alert threshold per (vehicle, person) pair to avoid repeated alerts
last_vehicle_person_alert = {}

while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Failed to read frame from stream — wait briefly and retry
            print("Warning: empty frame received from stream, retrying...")
            time.sleep(0.1)
            continue

        # --- Simple stabilization to reduce zoom/jitter from IP camera ---
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                # Ensure we have feature points to track
                if prev_pts is None or len(prev_pts) < 50:
                    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                                       minDistance=8, blockSize=7)

                if prev_pts is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
                    if p1 is not None and st is not None:
                        st = st.reshape(-1)
                        good_prev = prev_pts[st == 1]
                        good_curr = p1[st == 1]
                        if len(good_prev) >= 6:
                            # Estimate affine that maps current points to previous points (align current -> prev)
                            M, inliers = cv2.estimateAffinePartial2D(good_curr, good_prev, method=cv2.RANSAC,
                                                                    ransacReprojThreshold=3)
                            if M is not None:
                                h, w = frame.shape[:2]
                                # Warp the current frame to align it with the previous frame
                                frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
                                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)

            # Update previous frame and feature points
            prev_gray = gray.copy()
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                               minDistance=8, blockSize=7)
        except Exception as e:
            # If stabilization fails for any reason, continue without it
            # print(f"Stabilization warning: {e}")
            pass

        # Normalize frame size to a fixed resolution to prevent zooming effects
        try:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        except Exception:
            # If resize fails, proceed with the original frame
            pass

        # Submit a small copy for asynchronous detection (overwrite semantics)
        try:
            small = cv2.resize(frame, (DET_W, DET_H))
            with frame_lock:
                frame_for_detection = small
                new_frame_for_detection = True
        except Exception:
            pass

        # --- Recording: initialize writer (to a temp file) and rotate every CLIP_DURATION seconds ---
        try:
            now = time.time()
            
            # Measure actual frame rate from capture timing
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) >= 10:
                time_span = frame_times[-1] - frame_times[0]
                if time_span > 0:
                    measured_fps = max(5, min(60, (len(frame_times) - 1) / time_span))
            
            # Use measured FPS, fallback to 15 if detection fails
            fps_to_use = measured_fps if measured_fps > 0 else 15.0

            if video_writer is None:
                # Create writer with measured FPS using XVID codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(writer_tmp_path, fourcc, fps_to_use, (TARGET_W, TARGET_H))
                clip_start_time = now

            # Draw recording indicator and remaining time on the frame
            try:
                remaining = max(0, int(CLIP_DURATION - (now - (clip_start_time or now))))
                cv2.putText(frame, f"REC {remaining}s FPS:{measured_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.circle(frame, (80, 25), 8, (0, 0, 255), -1)
            except Exception:
                pass

            # Write the current frame to the active (temp) clip
            if video_writer is not None:
                try:
                    video_writer.write(frame)
                except Exception as e:
                    print(f"Error writing frame: {e}")

            # If current clip duration exceeded, finalize this temp file and atomically replace the public file
            if clip_start_time is not None and (now - clip_start_time) >= CLIP_DURATION:
                try:
                    video_writer.release()
                    print(f"Clip finalized: {measured_fps:.1f} FPS, {frame_count} frames")
                    # Atomically replace the visible output file
                    try:
                        if os.path.exists(OUT_FILENAME):
                            os.remove(OUT_FILENAME)
                        os.replace(writer_tmp_path, OUT_FILENAME)
                        print(f"Saved to {OUT_FILENAME}")
                    except Exception as e:
                        # Fallback: try rename
                        try:
                            os.rename(writer_tmp_path, OUT_FILENAME)
                            print(f"Saved to {OUT_FILENAME} (via rename)")
                        except Exception:
                            print(f"Failed to finalize file: {e}")
                except Exception as e:
                    print(f"Error during clip finalization: {e}")

                # Create a new writer (temp file) for the next clip
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(writer_tmp_path, fourcc, fps_to_use, (TARGET_W, TARGET_H))
                    clip_start_time = now
                    frame_count = 0
                except Exception as e:
                    print(f"Error creating new writer: {e}")
                    video_writer = None
                    clip_start_time = None
            
            frame_count += 1
        except Exception as e:
            print(f"Recording error: {e}")

        # Use latest async detections (may be None until detector produces first result)
        with results_lock:
            detections = latest_detections.copy() if latest_detections is not None else []

        nearest_object = None
        min_distance = float('inf')
        detected_objects = []

        # Collect persons and vehicles for pairwise alerts
        persons = []
        vehicles = []
        vehicle_types = ["car", "bus", "motorcycle"]

        for det in detections:
            label = det['label']
            cords = det['xyxy']
            colorGreen = (0, 255, 0)
            colorYellow = (0, 255, 255)
            colorBlue = (255, 0, 0)
            colorRed = (0, 0, 255)

            thickness = 2

            # compute distance using pixel width
            pixel_w = cords[2] - cords[0]
            distance = calculate_distance_from_pixels(pixel_w, frame.shape[1], label)
            cx = int((cords[0] + cords[2]) / 2)
            cy = int((cords[1] + cords[3]) / 2)

            if distance < min_distance:
                min_distance = distance
                nearest_object = (label, round(distance, 1), cords)
                detected_objects = [(label, round(distance, 1))]

            # THE CLOSEST RED OBJECT DOES NOT MATTER
            # HUMAN GREEN
            # CAR YELLOW
            # OTHERS ARE BLUE

            if label == "person":
                frame = blur_person(frame, cords)
                # Use red color and thicker line for very close humans (visual only, no alerts)
                if distance <= 1:
                    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorRed, thickness + 2)
                    cv2.putText(frame, f"WARNING! {label} - {distance:.1f}m", (cords[0], cords[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorRed, thickness + 1)
                else:
                    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorGreen, thickness)
                    cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorGreen, thickness)

                # Add to persons list (no direct alerts for persons)
                pid = f"person_{cords[0]}_{cords[1]}_{cords[2]}_{cords[3]}"
                persons.append({"id": pid, "distance": distance, "cords": cords, "cx": cx, "cy": cy, "label": label})
            elif label in vehicle_types:
                # Vehicle drawing
                cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorYellow, thickness)
                cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colorYellow, thickness)

                vid = f"{label}_{cords[0]}_{cords[1]}_{cords[2]}_{cords[3]}"
                vehicles.append({"id": vid, "distance": distance, "cords": cords, "cx": cx, "cy": cy, "label": label})
            elif label in class_avg_sizes:
                cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorBlue, thickness)
                cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colorBlue, thickness)

        # Pairwise vehicle -> person proximity alerts
        try:
            for v in vehicles:
                for p in persons:
                    inter = abs(v["distance"] - p["distance"])  # approximate distance between objects along depth
                    key = f"{v['id']}|{p['id']}"
                    if inter <= 3.0:
                        step_size = 0.1 if inter <= 1.0 else 0.5
                        # Use ceiling so alert triggers when crossing below the threshold step
                        threshold = math.ceil(inter / step_size) * step_size
                        # Clamp threshold to max 3.0
                        if threshold > 3.0:
                            threshold = 3.0

                        last = last_vehicle_person_alert.get(key)
                        if last is None or last != threshold:
                            # compute position of vehicle relative to person horizontally
                            if v['cx'] < p['cx'] - (frame.shape[1] // 10):
                                pos = "LEFT"
                            elif v['cx'] > p['cx'] + (frame.shape[1] // 10):
                                pos = "RIGHT"
                            else:
                                pos = "FORWARD"

                            # enqueue alert for vehicle approaching person
                            queue.put((v['label'], round(inter, 2), pos))
                            last_vehicle_person_alert[key] = threshold
                    else:
                        # If objects are farther apart, clear previous alert state for this pair
                        if key in last_vehicle_person_alert:
                            del last_vehicle_person_alert[key]
        except Exception:
            pass

        #  en yakın
        if nearest_object:

            if nearest_object[0] in class_avg_sizes:  # coordinats
                cv2.rectangle(frame, (nearest_object[2][0], nearest_object[2][1]),(nearest_object[2][2], nearest_object[2][3]), (0, 0, 255), thickness)
                text = f"{nearest_object[0]} - {round(nearest_object[1], 1)}m"
                cv2.putText(frame, text, (nearest_object[2][0], nearest_object[2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colorRed, thickness)

            if nearest_object[1] <= 12.5:  # give audio feedback if the distance is smaller or larger than the specified value
                # Only give immediate distance alerts for vehicles (not for persons)
                if nearest_object[0] in ["car", "bus", "motorcycle"]:
                    position = get_position(frame.shape[1], nearest_object[2]) #frame_width, box
                    queue.put((nearest_object[0], nearest_object[1], position))  # label, distance, position

            detected_objects.clear()
    else:
        frame = cap.retrieve()[1]



    cv2.imshow('Audio World ', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        pause = not pause

cap.release()
cv2.destroyAllWindows()
