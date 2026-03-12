import os
import time
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


# This video source is 1280x720 30FPS
source = os.getenv("VIDEO_STREAM")

capture_mode = cv2.CAP_FFMPEG
feed = cv2.VideoCapture(source, capture_mode)
prev = time.time()
vehicle_class_ids = {2, 3, 5, 7}
model = YOLO("yolov8n.pt")

while True:
    ok, frame = feed.read()

    if not ok:
        print("Failed to read frame.")
        time.sleep(0.5)
        feed.release()
        feed = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        continue
    
    height, width = frame.shape[:2] # y, x

    # TODO: FIX ROI SETTINGS
    roi_w, roi_h = 250, 150
    roi_zoom = 2
    roi_offset = 50

    x1 = (width - roi_w) // 2
    y1 = ((height - roi_h) // 2) + roi_offset
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    roi = frame[y1:y2, x1:x2]
    roi_up = cv2.resize(roi, (roi_w * roi_zoom, roi_h * roi_zoom),
                        interpolation=cv2.INTER_LINEAR)
    
    results = model.predict(roi_up, conf=0.25, verbose=False)  # conf threshold tweak here
    r = results[0]

    # r.boxes contains boxes on this frame
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            if cls_id not in vehicle_class_ids:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf.item())
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(roi_up, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(roi_up, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FPS display (rough)
    now = time.time()
    fps = 1.0 / max(1e-6, (now - prev))
    prev = now
    cv2.putText(roi_up, f"FPS {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Vehicle Detection", roi_up)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

feed.release()
cv2.destroyAllWindows()    