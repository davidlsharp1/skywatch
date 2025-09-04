import cv2
import datetime
import time
import os

# --- Setup ---
stream = cv2.VideoCapture(0)
fps = int(stream.get(cv2.CAP_PROP_FPS)) or 30
width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=15)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec

# --- Motion Recording Parameters ---
recording = False
last_motion_time = 0
motion_timeout = 3  # seconds to keep recording after motion stops

if not os.path.exists("captures"):
    os.makedirs("captures")

while True:
    ret, frame = stream.read()
    if not ret:
        break

    # Preprocessing
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # helps reduce noise
    mask = object_detector.apply(blurred)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # sensitive
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Start recording if motion detected
    if motion_detected:
        last_motion_time = time.time()
        if not recording:
            filename = datetime.datetime.now().strftime("captures/%Y-%m-%d_%H-%M-%S.mp4")
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            recording = True
            print(f"[INFO] Started recording: {filename}")

    # Write frame if recording
    if recording:
        out.write(frame)
        # Stop recording if motion has stopped for a while
        if time.time() - last_motion_time > motion_timeout:
            recording = False
            out.release()
            print("[INFO] Stopped recording")

    # Show preview
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

# Cleanup
stream.release()
if recording:
    out.release()
cv2.destroyAllWindows()
