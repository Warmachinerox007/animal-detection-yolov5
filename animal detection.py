# animal_detection.py

import torch
import cv2
import time
import pygetwindow as gw
import pyautogui

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define animals to detect
ANIMAL_CLASSES = [
    'cat', 'dog', 'bird', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

print("✅ Real-time animal detection started. Press 'q' to quit.")

# Window setup
cv2.namedWindow("🐾 Real-Time Animal Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🐾 Real-Time Animal Detection", 800, 600)

# Allow time for window to create
time.sleep(1)

# Bring window to front
try:
    win = gw.getWindowsWithTitle("🐾 Real-Time Animal Detection")[0]
    win.restore()
    win.activate()
    pyautogui.moveTo(win.left + 10, win.top + 10)
except Exception as e:
    print("⚠️ Could not focus window:", e)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed.")
        break

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection
    results = model(img_rgb)
    detections = results.pandas().xyxy[0]
    animals = detections[detections['name'].isin(ANIMAL_CLASSES)]

    # Draw boxes
    for _, row in animals.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show live detection
    cv2.imshow("🐾 Real-Time Animal Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
