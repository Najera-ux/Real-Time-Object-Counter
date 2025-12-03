from ultralytics import YOLO
import cv2

# --- Load YOLO model ---
model = YOLO("yolov8n.pt")   # smallest YOLO model

# --- Open webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Run YOLO on frame ---
    results = model(frame, verbose=False)

    # --- Count objects ---
    count = 0
    for r in results:
        for box in r.boxes:
            count += 1

            # Draw box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # --- Show object count ---
    cv2.putText(frame, f"Count: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # --- Display ---
    cv2.imshow("YOLO Object Counter", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
