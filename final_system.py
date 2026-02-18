import cv2
import numpy as np
import time

# ---------------------------------------
# Load YOLO
# ---------------------------------------
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ---------------------------------------
# Load Haar Face Detector
# ---------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------
# Webcam
# ---------------------------------------
cap = cv2.VideoCapture(0)

# ---------------------------------------
# Behavior Tracking
# ---------------------------------------
suspicious_start_time = None
SUSPICIOUS_THRESHOLD = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # ---------------------------------------
    # YOLO Object Detection
    # ---------------------------------------
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416),
        (0, 0, 0), True, crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_count = 0
    phone_detected = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]

            if label == "person":
                person_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if label == "cell phone":
                phone_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # ---------------------------------------
    # Face Detection + Direction
    # ---------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_count = len(faces)
    looking_direction = "Forward"

    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Horizontal
        if face_center_x < width // 3:
            looking_direction = "Left"
        elif face_center_x > 2 * width // 3:
            looking_direction = "Right"

        # Vertical
        elif face_center_y > 2 * height // 3:
            looking_direction = "Down"
        elif face_center_y < height // 3:
            looking_direction = "Up"
        else:
            looking_direction = "Forward"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # ---------------------------------------
    # Decision Logic with Stability
    # ---------------------------------------
    current_time = time.time()
    suspicious_now = False
    status_text = "Normal"

    if phone_detected:
        status_text = "Phone Detected - Cheating"
        suspicious_start_time = None

    elif person_count > 1:
        status_text = "Multiple Persons - Cheating"
        suspicious_start_time = None

    elif face_count == 0:
        suspicious_now = True
        status_text = "No Face - Suspicious"

    elif looking_direction != "Forward":
        suspicious_now = True
        status_text = f"Looking {looking_direction} - Suspicious"

    else:
        suspicious_start_time = None
        status_text = "Normal"

    # Timer Logic (avoid instant false alerts)
    if suspicious_now:
        if suspicious_start_time is None:
            suspicious_start_time = current_time
        elif current_time - suspicious_start_time > SUSPICIOUS_THRESHOLD:
            status_text = "Confirmed Suspicious Behavior"

    # ---------------------------------------
    # Display
    # ---------------------------------------
    cv2.putText(
        frame,
        status_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Integrated Cheating Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import time

# # ==============================
# # LOAD YOLO (unchanged)
# # ==============================
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # ==============================
# # HAAR FACE
# # ==============================
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# # ==============================
# # CAMERA SETTINGS (smooth)
# # ==============================
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# # ==============================
# # STABILITY SETTINGS
# # ==============================
# direction_buffer = []
# BUFFER_SIZE = 6
# SUSPICIOUS_THRESHOLD = 2
# suspicious_start_time = None

# frame_count = 0
# YOLO_INTERVAL = 5   # run YOLO every 5 frames

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     height, width = frame.shape[:2]

#     phone_detected = False
#     person_count = 0

#     # ==============================
#     # RUN YOLO ONLY SOMETIMES (reduce lag)
#     # ==============================
#     if frame_count % YOLO_INTERVAL == 0:
#         blob = cv2.dnn.blobFromImage(
#             frame, 0.00392, (320, 320),
#             (0, 0, 0), True, crop=False
#         )

#         net.setInput(blob)
#         outputs = net.forward(output_layers)

#         boxes = []
#         confidences = []
#         class_ids = []

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > 0.5:
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         for i in range(len(boxes)):
#             if i in indexes:
#                 x, y, w, h = boxes[i]
#                 label = classes[class_ids[i]]

#                 if label == "person":
#                     person_count += 1
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

#                 if label == "cell phone":
#                     phone_detected = True
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,255), 2)

#     # ==============================
#     # FACE DETECTION
#     # ==============================
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.2, 5)

#     face_count = len(faces)
#     looking_direction = "Forward"

#     if face_count == 1:
#         x, y, w, h = faces[0]
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

#         face_center_x = x + w//2
#         face_center_y = y + h//2

#         # ==============================
#         # NEW LOGIC (no bias)
#         # ==============================

#         left_bound = x + w * 0.35
#         right_bound = x + w * 0.65
#         down_bound = y + h * 0.70

#         if face_center_x < left_bound:
#             looking_direction = "Left"
#         elif face_center_x > right_bound:
#             looking_direction = "Right"
#         elif face_center_y > down_bound:
#             looking_direction = "Down"
#         else:
#             looking_direction = "Forward"

#         direction_buffer.append(looking_direction)

#         if len(direction_buffer) > BUFFER_SIZE:
#             direction_buffer.pop(0)

#         final_direction = max(set(direction_buffer), key=direction_buffer.count)

#     else:
#         final_direction = "NoFace"

#     # ==============================
#     # DECISION LOGIC
#     # ==============================
#     current_time = time.time()
#     suspicious_now = False
#     status_text = "Normal"

#     if phone_detected:
#         status_text = "Phone Detected - Cheating"
#         suspicious_start_time = None

#     elif person_count > 1:
#         status_text = "Multiple Persons - Cheating"
#         suspicious_start_time = None

#     elif face_count == 0:
#         suspicious_now = True
#         status_text = "No Face - Suspicious"

#     elif final_direction in ["Left", "Right", "Down"]:
#         suspicious_now = True
#         status_text = f"Looking {final_direction} - Suspicious"

#     else:
#         suspicious_start_time = None
#         status_text = "Normal"

#     if suspicious_now:
#         if suspicious_start_time is None:
#             suspicious_start_time = current_time
#         elif current_time - suspicious_start_time > SUSPICIOUS_THRESHOLD:
#             status_text = "Confirmed Suspicious Behavior"

#     # ==============================
#     # DISPLAY
#     # ==============================
#     cv2.putText(
#         frame,
#         status_text,
#         (10, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 0, 255),
#         2
#     )

#     cv2.imshow("Integrated Cheating Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()