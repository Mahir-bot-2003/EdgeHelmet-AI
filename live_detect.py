from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/helmet_combined_v2/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Starting live helmet detection from webcam...")
print("[INFO] Press 'q' to quit")
print("[INFO] Press 's' to save a screenshot")

frame_count = 0
helmet_total = 0
no_helmet_total = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Failed to read frame from camera")
        break
    
    frame_count += 1
    
    # Run YOLO detection
    results = model(frame, conf=0.5)
    
    # Get detections
    detections = results[0].boxes.data.cpu().numpy()
    
    helmet_in_frame = 0
    no_helmet_in_frame = 0
    
    # Draw bounding boxes for each detection
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        # Determine color and label
        if class_id == 0:  # Helmet detected
            color = (0, 255, 0)  # Green
            label = f"Helmet {conf:.2f}"
            helmet_in_frame += 1
            helmet_total += 1
        else:  # No helmet detected
            color = (0, 0, 255)  # Red
            label = f"No Helmet {conf:.2f}"
            no_helmet_in_frame += 1
            no_helmet_total += 1
        
        # Draw rectangle around detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 8), 
                      (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw status bar at top
    status_text = f"Frame: {frame_count} | This Frame - Helmet: {helmet_in_frame} | No Helmet: {no_helmet_in_frame}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw summary at bottom
    if no_helmet_in_frame == 0:
        alert_text = "✓ ALL SAFE - Everyone wearing helmets"
        alert_color = (0, 255, 0)  # Green
    else:
        alert_text = f"⚠ ALERT - {no_helmet_in_frame} person(s) without helmet!"
        alert_color = (0, 0, 255)  # Red
    
    cv2.rectangle(frame, (0, frame.shape[0] - 35), (frame.shape[1], frame.shape[0]), 
                  (0, 0, 0), -1)
    cv2.putText(frame, alert_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
    
    # Display FPS
    fps_text = f"FPS: ~30"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
cv2.destroyAllWindows()

print(f"\n[INFO] Detection Summary:")
print(f"  Total frames processed: {frame_count}")
print(f"  Total helmet detections: {helmet_total}")
print(f"  Total no-helmet detections: {no_helmet_total}")