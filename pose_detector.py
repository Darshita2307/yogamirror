import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark

            # Knee angle (left leg)
            hip = [lm[23].x, lm[23].y]
            knee = [lm[25].x, lm[25].y]
            ankle = [lm[27].x, lm[27].y]

            angle = calculate_angle(hip, knee, ankle)
            print(f"Left Knee Angle: {angle}")

            cv2.putText(frame, f"Knee: {angle}",
                       (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)

        cv2.imshow("YogaMirror", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()