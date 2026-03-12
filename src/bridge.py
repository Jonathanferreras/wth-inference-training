import os
import time
import cv2
from dotenv import load_dotenv

load_dotenv()

# This video source is 1280x720 30FPS
source = os.getenv("VIDEO_STREAM")

capture_mode = cv2.CAP_FFMPEG
feed = cv2.VideoCapture(source, capture_mode)

while True:
    ok, frame = feed.read()

    if not ok:
        print("Failed to read frame.")
        time.sleep(0.5)
        feed.release()
        feed = cv2.VideoCapture(source, capture_mode)
        continue

    height, width = frame.shape[:2]
    fps = feed.get(cv2.CAP_PROP_FPS)
    print(f"Width: {width}, Height: {height}, FPS: {fps}")

    roi_w, roi_h = 250, 150
    roi_zoom = 2
    roi_x_offset = 0
    roi_y_offset = 175

    x1 = ((width - roi_w) // 2) + roi_x_offset
    y1 = ((height - roi_h) // 2) + roi_y_offset
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    roi = frame[y1:y2, x1:x2]
    
    # process roi
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    roi_edges = cv2.Canny(roi_gray_blur, 50, 150)
    
    edges_display = cv2.resize(
        roi_edges, 
        (roi_w * roi_zoom, roi_h * roi_zoom)
    )    
    roi_display = cv2.resize(
        roi_gray,
        (roi_w * roi_zoom, roi_h * roi_zoom),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    cv2.imshow("Edges", edges_display)
    cv2.imshow("ROI", roi_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

feed.release()
cv2.destroyAllWindows()