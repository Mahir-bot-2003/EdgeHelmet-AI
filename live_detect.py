from ultralytics import YOLO
import cv2
import time
from collections import deque
import numpy as np

# Load your trained model
model = YOLO("runs/detect/helmet_combined_v2/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution
# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get video properties for recording
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0 or fps > 60:
    fps = 20  # Safe fallback for webcam estimation

# Setup saving directory and auto-increment folder name
import os
base_dir = "runs/detect/live_webcam_recording"
save_dir = base_dir
counter = 2
while os.path.exists(save_dir):
    save_dir = f"{base_dir}{counter}"
    counter += 1
os.makedirs(save_dir, exist_ok=True)

# Setup VideoWriter
output_path = os.path.join(save_dir, "webcam_output.mp4")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"[INFO] Saving live feed recording to: {output_path}")
print("[INFO] Starting live helmet detection with temporal smoothing...")
print("[INFO] Press 'q' to quit")
print("[INFO] Press 's' to save a screenshot")

# ============================================================
# TEMPORAL SMOOTHING CONFIG
# ============================================================
SMOOTHING_WINDOW = 30   # ~1 second of history at 30fps
SWITCH_THRESHOLD = 0.7  # 70% of frames must agree before label switches (prevents flicker)
IOU_THRESHOLD = 0.3     # Minimum overlap to consider it the same person
CONF_THRESHOLD = 0.4    # Lower threshold — smoothing handles the noise

# ============================================================
# TRACKED PERSON CLASS
# ============================================================
class TrackedPerson:
    """Tracks a single person across frames using bounding box overlap."""
    def __init__(self, bbox, class_id, conf):
        self.bbox = bbox                          # (x1, y1, x2, y2)
        self.history = deque(maxlen=SMOOTHING_WINDOW)  # stores class_id per frame
        self.history.append(class_id)
        self.last_conf = conf
        self.frames_missing = 0                   # how many frames since last seen

    def update(self, bbox, class_id, conf):
        self.bbox = bbox
        self.history.append(class_id)
        self.last_conf = conf
        self.frames_missing = 0

    def get_smoothed_class(self):
        """Supermajority vote — requires 70% agreement to switch label."""
        if len(self.history) == 0:
            return 0
        total = len(self.history)
        helmet_count = sum(1 for c in self.history if c == 0)
        no_helmet_ratio = (total - helmet_count) / total
        helmet_ratio = helmet_count / total
        # Only label as "No Helmet" if 70%+ of recent frames say so
        # Otherwise default to "Helmet" (safer assumption)
        if no_helmet_ratio >= SWITCH_THRESHOLD:
            return 1
        elif helmet_ratio >= SWITCH_THRESHOLD:
            return 0
        else:
            # Not enough agreement — keep previous label (default to helmet for safety)
            return self._prev_class if hasattr(self, '_prev_class') else 0

    def _update_prev_class(self):
        self._prev_class = self.get_smoothed_class()

    def get_confidence_text(self):
        """Show vote ratio as a stability indicator."""
        total = len(self.history)
        smoothed_class = self.get_smoothed_class()
        votes_for = sum(1 for c in self.history if c == smoothed_class)
        return f"{votes_for}/{total}"


def compute_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# ============================================================
# MAIN LOOP
# ============================================================
tracked_persons = []  # List of TrackedPerson objects
frame_count = 0
helmet_total = 0
no_helmet_total = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read frame from camera")
        break

    frame_count += 1

    # Run YOLO detection
    results = model(frame, conf=CONF_THRESHOLD)
    detections = results[0].boxes.data.cpu().numpy()

    # ---- Match current detections to tracked persons ----
    current_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        current_detections.append({
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'class_id': int(class_id),
            'conf': float(conf),
            'matched': False
        })

    # Try to match each detection to an existing tracked person
    for person in tracked_persons:
        best_iou = 0
        best_det_idx = -1
        for i, det in enumerate(current_detections):
            if det['matched']:
                continue
            iou = compute_iou(person.bbox, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det_idx = i

        if best_iou >= IOU_THRESHOLD and best_det_idx >= 0:
            det = current_detections[best_det_idx]
            person.update(det['bbox'], det['class_id'], det['conf'])
            det['matched'] = True
        else:
            person.frames_missing += 1

    # Create new tracked persons for unmatched detections
    for det in current_detections:
        if not det['matched']:
            tracked_persons.append(
                TrackedPerson(det['bbox'], det['class_id'], det['conf'])
            )

    # Remove persons not seen for a while
    tracked_persons = [p for p in tracked_persons if p.frames_missing <= 10]

    # ---- Draw results using SMOOTHED labels ----
    helmet_in_frame = 0
    no_helmet_in_frame = 0

    for person in tracked_persons:
        if person.frames_missing > 0:
            continue  # Don't draw persons not visible in this frame

        x1, y1, x2, y2 = person.bbox
        smoothed_class = person.get_smoothed_class()
        stability = person.get_confidence_text()

        if smoothed_class == 0:  # Helmet
            color = (0, 255, 0)
            label = f"Helmet ({stability})"
            helmet_in_frame += 1
            helmet_total += 1
        else:  # No helmet
            color = (0, 0, 255)
            label = f"No Helmet ({stability})"
            no_helmet_in_frame += 1
            no_helmet_total += 1

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 8),
                      (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        person._update_prev_class()

    # ---- Calculate actual FPS ----
    curr_time = time.time()
    actual_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # ---- Draw status bar at top ----
    status_text = f"Frame: {frame_count} | Helmet: {helmet_in_frame} | No Helmet: {no_helmet_in_frame} | Smoothing: {SMOOTHING_WINDOW}f"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    fps_text = f"FPS: {actual_fps:.0f}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---- Draw alert bar at bottom ----
    if no_helmet_in_frame == 0:
        alert_text = "ALL SAFE - Everyone wearing helmets"
        alert_color = (0, 255, 0)
    else:
        alert_text = f"ALERT - {no_helmet_in_frame} person(s) without helmet!"
        alert_color = (0, 0, 255)

    cv2.rectangle(frame, (0, frame.shape[0] - 35), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), -1)
    cv2.putText(frame, alert_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)

    # Write frame to the video output
    out.write(frame)

    # Show the frame
    cv2.imshow("Live Helmet Detection - Webcam Feed", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[INFO] Exiting...")
        break
    elif key == ord('s'):
        filename = f"screenshot_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved: {filename}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n[INFO] Detection Summary:")
print(f"  Total frames processed: {frame_count}")
print(f"  Total helmet detections: {helmet_total}")
print(f"  Total no-helmet detections: {no_helmet_total}")
print(f"  Live video saved to: {output_path}")
