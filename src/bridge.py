import os
import time
import cv2
import numpy as np
from dotenv import load_dotenv
from src.dataset_imports import march_19_videos

load_dotenv()

vid_count = 0
capture_mode = cv2.CAP_FFMPEG
# source = march_11_videos
# source = march_18_videos
source = march_19_videos
feed = cv2.VideoCapture(source[vid_count], capture_mode)

# This video source is 1280x720 30FPS
# source = os.getenv("VIDEO_STREAM")
# feed = cv2.VideoCapture(source, capture_mode)

while True:
    ok, frame = feed.read()

    if not ok and vid_count < len(source) - 1:
        print("Failed to read frame.")
        time.sleep(0.5)
        feed.release()
        vid_count += 1
        feed = cv2.VideoCapture(source[vid_count], capture_mode)
        # feed = cv2.VideoCapture(source, capture_mode)
        continue
    elif not ok:
        print("No more frames available.")
        break

    height, width = frame.shape[:2]
    fps = feed.get(cv2.CAP_PROP_FPS)
    # print(f"Width: {width}, Height: {height}, FPS: {fps}")

    # good for datasets before 03-18-2026
    # roi_w, roi_h = 165, 165
    # roi_zoom = 2
    # roi_x_offset = 25
    # roi_y_offset = 135

    roi_w, roi_h = 165, 135
    roi_zoom = 2
    roi_x_offset = -105
    # roi_y_offset = 85
    roi_y_offset = 105


    x1 = ((width - roi_w) // 2) + roi_x_offset
    y1 = ((height - roi_h) // 2) + roi_y_offset
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    roi = frame[y1:y2, x1:x2]

    # process roi
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    roi_edges = cv2.Canny(roi_gray, 40, 165)
    # roi_edges = cv2.Canny(roi_gray_blur, 40, 165)


    # Hough line detection
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=10,
    )

    roi_lines_display = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)

    best_line = None
    best_length = 0
    best_angle = None

    if lines is not None:
        for line in lines:
            x1l, y1l, x2l, y2l = line[0]

            dx = x2l - x1l
            dy = y2l - y1l
            length = np.hypot(dx, dy)

            angle_deg = np.degrees(np.arctan2(dy, dx))
            angle_deg = abs(angle_deg)
            
            if angle_deg > 90:
                angle_deg = 180 - angle_deg

            mid_y = (y1l + y2l) / 2

            # draw all raw lines in green
            cv2.line(roi_lines_display, (x1l, y1l), (x2l, y2l), (0, 255, 0), 1)

            # basic filters
            if length < 35:
                continue

            if mid_y < roi_h * 0.35:
                continue

            if length > best_length:
                best_length = length
                best_line = (x1l, y1l, x2l, y2l)
                best_angle = angle_deg

    # highlight selected line
    if best_line is not None:
        x1l, y1l, x2l, y2l = best_line
        cv2.line(roi_lines_display, (x1l, y1l), (x2l, y2l), (0, 0, 255), 2)

        cv2.putText(
            roi_lines_display,
            f"Angle: {best_angle:.1f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        print(f"Selected line angle: {best_angle:.2f} deg, length: {best_length:.2f}")
    else:
        print("No valid bridge line found.")

    edges_display = cv2.resize(
        roi_edges, (roi_w * roi_zoom, roi_h * roi_zoom)
    )

    roi_display = cv2.resize(
        roi_gray, (roi_w * roi_zoom, roi_h * roi_zoom),
        interpolation=cv2.INTER_LINEAR
    )

    lines_display = cv2.resize(
        roi_lines_display, (roi_w * roi_zoom, roi_h * roi_zoom),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    # cv2.imshow("Edges", edges_display)
    cv2.imshow("ROI", roi_display)
    cv2.imshow("Hough Lines", lines_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

feed.release()
cv2.destroyAllWindows()
