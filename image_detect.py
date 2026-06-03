from ultralytics import YOLO
import cv2
import os
model = YOLO("runs/detect/helmet_combined_v2/weights/best.pt")
image_path = "test.jpg"  
results = model(image_path)
annotated = results[0].plot()
cv2.imshow("Helmet Detection", annotated)
cv2.waitKey(0)  
cv2.destroyAllWindows()
cv2.imwrite("result_" + os.path.basename(image_path), annotated)
print(f"Result saved as: result_{os.path.basename(image_path)}")
