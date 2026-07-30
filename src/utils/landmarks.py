# src/utils/landmarks.py

import numpy as np
import mediapipe as mp

from src import config

mp_hands = mp.solutions.hands


class HandLandmarkExtractor:
    """
    Wraps MediaPipe Hands to extract a normalized, fixed-shape feature vector
    per frame: 128-dim = 2 hands x (21 landmarks x 3 coords + 1 presence flag).
    """

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process_frame(self, frame_bgr):
        """
        Takes a single BGR frame (as read by OpenCV).
        Returns: (feature_vector: np.ndarray shape (128,), annotated_frame for display)
        """
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_vec = np.zeros(config.FEATURE_DIM_PER_HAND, dtype=np.float32)
        right_vec = np.zeros(config.FEATURE_DIM_PER_HAND, dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                vec = self._landmarks_to_vector(hand_landmarks)

                if label == "Left":
                    left_vec = vec
                else:
                    right_vec = vec

        feature_vector = np.concatenate([left_vec, right_vec])  # shape (128,)
        return feature_vector, results

    def _landmarks_to_vector(self, hand_landmarks):
        """
        Converts MediaPipe's 21 landmarks into a normalized 64-dim vector:
        63 coords (translated + scale-normalized) + 1 presence flag.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )  # shape (21, 3)

        # Translate: make wrist (landmark 0) the origin
        wrist = coords[0].copy()
        coords -= wrist

        # Scale-normalize: divide by wrist-to-middle-finger-MCP distance (landmark 9)
        ref_distance = np.linalg.norm(coords[9])
        if ref_distance > 1e-6:
            coords /= ref_distance

        flat_coords = coords.flatten()  # shape (63,)
        presence_flag = np.array([1.0], dtype=np.float32)

        return np.concatenate([flat_coords, presence_flag])  # shape (64,)

    def close(self):
        self.hands.close()
        
if __name__ == "__main__":
    import cv2

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        feature_vector, results = extractor.process_frame(frame)
        print(feature_vector.shape, "| nonzero:", np.count_nonzero(feature_vector))

        cv2.imshow("Landmark Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()