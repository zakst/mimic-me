import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(image_rgb)
    pose_results = pose.process(image_rgb)

    image_bgr = image.copy()

    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(image_bgr, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand + Arm Tracking", image_bgr)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
