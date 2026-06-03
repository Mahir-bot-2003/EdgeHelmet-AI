from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs/detect/helmet_combined_v2/weights/best.pt")

# Open video
cap = cv2.VideoCapture("test5.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output folder (auto-increment: predict, predict2, predict3...)
base_dir = "runs/detect/predict"
save_dir = base_dir
counter = 2
while os.path.exists(save_dir):
    save_dir = f"{base_dir}{counter}"
    counter += 1
os.makedirs(save_dir, exist_ok=True)

# Setup video writer
output_path = os.path.join(save_dir, "output.mp4")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Saving output to: {output_path}")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame to output video
    out.write(annotated_frame)

    # Show live processing
    cv2.imshow("Helmet Detection", annotated_frame)

    # Wait for the correct amount of time to match original video speed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to: {output_path}")
