# SMART-VOLUME-CONTROLLER-USING-HAND-GESTURES-AND-OPENCV-PYTHON
Developed a **gesture-based volume control system** using **Python, OpenCV, MediaPipe, and Pycaw** to enable touchless audio control. The system tracks **hand movements in real-time**, detects the distance between the thumb and index finger, and adjusts system volume accordingly.
import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Initialize MediaPipe Hand Tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volMin, volMax = volume.GetVolumeRange()[:2]
    print("Pycaw initialized successfully.")
except Exception as e:
    print("Error: Unable to initialize Pycaw. Make sure Pycaw is installed and you're on Windows.")
    print(e)
    exit()

# Calculate distance between thumb and index fingers
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read a frame from the camera.")
        break

    # Flip the frame for a mirrored view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # List to store landmarks
    lmList = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mpDraw.draw_landmarks(frame, hand_landmark, mpHands.HAND_CONNECTIONS)

    if lmList:
        # Get positions for thumb and index fingers
        thumb_tip = lmList[4]  # Thumb tip (landmark 4)
        index_tip = lmList[8]  # Index finger tip (landmark 8)

        # Draw circles and a line between the thumb and index finger
        cv2.circle(frame, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, index_tip, 10, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, thumb_tip, index_tip, (255, 0, 0), 3)

        # Calculate distance and adjust volume
        distance = calculate_distance(thumb_tip, index_tip)
        volume_level = np.interp(distance, [15, 220], [volMin, volMax])
        volume.SetMasterVolumeLevel(volume_level, None)

        # Display the volume bar and percentage
        volBar = np.interp(distance, [15, 220], [400, 150])
        volPercent = np.interp(distance, [15, 220], [0, 100])

        cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
        cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f'{int(volPercent)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Show the camera feed with the volume bar and hand landmarks
    cv2.imshow("Gesture Volume Control", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close the window
cap.release()
cv2.destroyAllWindows()

