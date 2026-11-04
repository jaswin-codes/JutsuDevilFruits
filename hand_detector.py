import math
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        all_hands_landmarks = []

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )
                all_hands_landmarks.append(hand_landmarks)

        return frame, all_hands_landmarks

    def get_landmark_list(self, hand_landmarks, frame_shape):
        h, w, _ = frame_shape
        landmark_list = []

        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append((cx, cy))

        return landmark_list

    def get_finger_states(self, landmark_list):
        """
        Returns [thumb, index, middle, ring, pinky]
        1 = up, 0 = down
        """
        if len(landmark_list) < 21:
            return [0, 0, 0, 0, 0]

        # Distance-based, so it works however the hand is rotated or which hand it is.
        def dist(a, b):
            return math.dist(landmark_list[a], landmark_list[b])

        # Thumb: extended if its tip (4) is further from the pinky base (17) than its IP joint (3)
        fingers = [1 if dist(4, 17) > dist(3, 17) else 0]

        # Four fingers: extended if the tip is further from the wrist (0) than the PIP joint;
        # a curled finger folds its tip back toward the palm.
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            fingers.append(1 if dist(tip, 0) > dist(pip, 0) else 0)

        return fingers