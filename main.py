from ultralytics import YOLO
import random
import cv2
import numpy as np

# Load YOLO model
model = YOLO("line_1.pt")

# Open a video capture object (you can change the index to 0 for the default webcam)
cap = cv2.VideoCapture("sample_1.mp4")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.5

while True:
    # Read a frame from the video source
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]

    for result in results:
        # Check if there are detections in the result
        if result.masks is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                color_number = classes_ids.index(int(box.cls[0]))
                cv2.fillPoly(frame, points, colors[color_number])

    # Display the result
    cv2.imshow("Real-time Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()