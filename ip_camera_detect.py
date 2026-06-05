from ultralytics import YOLO
import cv2
import time
import os

# ============================================================
#  PASTE YOUR CAMERA URL BELOW (ask your team for this)
# ============================================================
# Examples:
#   Hikvision:  "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
#   Dahua:      "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
#   HTTP/MJPEG: "http://192.168.1.100:8080/video"
#   Your webcam (for testing): 0

CAMERA_URL = 0  # <-- Replace 0 with your camera URL string when ready

# ============================================================

# Load your trained model
model = YOLO("runs/detect/helmet_combined_v2/weights/best.pt")

# Try to connect to the camera
print(f"[INFO] Connecting to camera: {CAMERA_URL}")
cap = cv2.VideoCapture(CAMERA_URL)

# For RTSP streams, set buffer size to 1 so we always get the latest frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Cannot connect to camera! Check the URL, credentials, and network.")
    print("        Make sure the camera is on the same network as this PC.")
    exit(1)

print("[INFO] Connected successfully!")
print("[INFO] Press 'q' to quit")
print("[INFO] Press 's' to save a screenshot")

# Video recording setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 25  # Default for IP cameras that don't report FPS

# Save recording to runs/detect/
save_dir = "runs/detect/ip_camera_recording"
os.makedirs(save_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(save_dir, f"recording_{timestamp}.mp4")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
print(f"[INFO] Recording to: {output_path}")

frame_count = 0
helmet_total = 0
no_helmet_total = 0
prev_time = time.time()
reconnect_attempts = 0

while True:
    ret, frame = cap.read()

    # Auto-reconnect if connection drops
    if not ret:
        reconnect_attempts += 1
        if reconnect_attempts > 10:
            print("[ERROR] Lost connection to camera after 10 retries. Exiting.")
            break
        print(f"[WARN] Connection lost. Reconnecting... (attempt {reconnect_attempts})")
        time.sleep(2)
        cap.release()
        cap = cv2.VideoCapture(CAMERA_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    reconnect_attempts = 0  # Reset on successful read
    frame_count += 1

    # Run YOLO detection
    results = model(frame, conf=0.5)

    # Get detections
    detections = results[0].boxes.data.cpu().numpy()

    helmet_in_frame = 0
    no_helmet_in_frame = 0

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        if class_id == 0:  # Helmet
            color = (0, 255, 0)
            label = f"Helmet {conf:.2f}"
            helmet_in_frame += 1
            helmet_total += 1
        else:  # No helmet
            color = (0, 0, 255)
            label = f"No Helmet {conf:.2f}"
            no_helmet_in_frame += 1
            no_helmet_total += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 8),
                      (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate actual FPS
    curr_time = time.time()
    actual_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Status bar
    status_text = f"Frame: {frame_count} | Helmet: {helmet_in_frame} | No Helmet: {no_helmet_in_frame}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    fps_text = f"FPS: {actual_fps:.0f}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Alert bar
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

    # Save frame to recording
    out.write(frame)

    # Display
    cv2.imshow("IP Camera - Helmet Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[INFO] Exiting...")
        break
    elif key == ord('s'):
        filename = f"screenshot_{timestamp}_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved: {filename}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n[INFO] Detection Summary:")
print(f"  Total frames: {frame_count}")
print(f"  Helmet detections: {helmet_total}")
print(f"  No-helmet detections: {no_helmet_total}")
print(f"  Recording saved: {output_path}")
