# import cv2
# import dlib

# # Initialize dlib's face detector and the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Start the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     num_faces = len(faces)

#     # Logic to determine if the scenario might be cheating
#     if num_faces == 0:
#         cheat_status = "No face detected - possible cheating"
#     elif num_faces > 1:
#         cheat_status = "Multiple faces detected - possible cheating"
#     else:
#         # Handle the case where exactly one face is detected
#         cheat_status = "One face detected - analyzing direction"
#         face = faces[0]
#         x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

#         # Facial landmarks
#         landmarks = predictor(gray, face)
#         nose_tip = landmarks.part(30).x
#         left_eye = sum([landmarks.part(n).x for n in range(36, 42)]) // 6
#         right_eye = sum([landmarks.part(n).x for n in range(42, 48)]) // 6

#         # Determine the direction the face is looking
#         if nose_tip < left_eye:
#             direction = "Looking Left"
#         elif nose_tip > right_eye:
#             direction = "Looking Right"
#         else:
#             direction = "Looking Forward"
        
#         cheat_status += f" - {direction}"

#     # Draw this status on the frame
#     cv2.putText(frame, cheat_status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import time

# ---------------------------------------
# LOAD YOLO (Phone + Person only)
# ---------------------------------------
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ---------------------------------------
# FACE DETECTOR (Haar - Fast & Stable)
# ---------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------
# WEBCAM SETTINGS (Reduce Lag)
# ---------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------------------------------
# STABILITY VARIABLES
# ---------------------------------------
suspicious_start = None
SUSPICIOUS_TIME = 2
frame_count = 0
YOLO_SKIP = 3  # run YOLO every 3 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width, _ = frame.shape

    phone_detected = False
    person_count = 0

    # ---------------------------------------
    # YOLO (SKIP FRAMES FOR SMOOTHNESS)
    # ---------------------------------------
    if frame_count % YOLO_SKIP == 0:
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416),
            (0, 0, 0), swapRB=True, crop=False
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

                if confidence > 0.6:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

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
    # FACE DETECTION (FAST)
    # ---------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    face_count = len(faces)
    looking_direction = "Forward"

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # ---- IMPROVED DIRECTION LOGIC ----
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # relative position inside bounding box
        x_ratio = (face_center_x - x) / w
        y_ratio = (face_center_y - y) / h

        if x_ratio < 0.35:
            looking_direction = "Left"
        elif x_ratio > 0.65:
            looking_direction = "Right"
        elif y_ratio > 0.65:
            looking_direction = "Down"
        else:
            looking_direction = "Forward"

    # ---------------------------------------
    # DECISION LOGIC
    # ---------------------------------------
    suspicious = False
    status = "Normal"

    if phone_detected:
        status = "Phone Detected - Cheating"
        suspicious_start = None

    elif person_count > 1:
        status = "Multiple Persons - Cheating"
        suspicious_start = None

    elif face_count == 0:
        suspicious = True
        status = "No Face - Suspicious"

    elif looking_direction != "Forward":
        suspicious = True
        status = f"Looking {looking_direction} - Suspicious"

    else:
        suspicious_start = None
        status = "Normal"

    # Stability Timer
    if suspicious:
        if suspicious_start is None:
            suspicious_start = time.time()
        elif time.time() - suspicious_start > SUSPICIOUS_TIME:
            status = "Confirmed Suspicious Behavior"

    # ---------------------------------------
    # DISPLAY
    # ---------------------------------------
    cv2.putText(
        frame,
        status,
        (10, 35),
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