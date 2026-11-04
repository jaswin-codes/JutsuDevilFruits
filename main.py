import argparse
import time
import cv2
import numpy as np
from hand_detector import HandDetector
from signs import match_sign, combine_signs
from stabilizer import SignStabilizer
from sequence import SequenceDetector
from combos import match_combo, MAX_COMBO_LENGTH
from effects import ParticleSystem, play_sound

def main():
    parser = argparse.ArgumentParser(description="Jutsu Code - hand-sign jutsu recognition")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector   = HandDetector(max_hands=2)
    stabilizer = SignStabilizer(buffer_size=10, confirm_threshold=7)
    sequencer  = SequenceDetector(max_length=MAX_COMBO_LENGTH)

    active_effect = None   # combo dict from combos.py while its effect plays
    effect_start  = 0.0
    EFFECT_SECONDS = 2.0
    particles = ParticleSystem()
    last_tick = time.monotonic()
    MISS_LIMIT = 15        # frames without a hand (~0.5s) before the combo is wiped
    missed_frames = 0

    print("Jutsu Code")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            break

        frame = cv2.flip(frame, 1)
        frame, hands = detector.find_hands(frame)

        if hands:
            missed_frames = 0
            # Order hands left-to-right on screen: MediaPipe's list order isn't stable
            hand_points = sorted((detector.get_landmark_list(h, frame.shape) for h in hands),
                                 key=lambda pts: pts[0][0])
            hand_fingers = [detector.get_finger_states(pts) for pts in hand_points]
            fingers   = hand_fingers[0]   # finger panel shows the left-most hand
            raw_sign  = combine_signs([match_sign(f) for f in hand_fingers])
            stable_sign = stabilizer.update(raw_sign)

            cv2.putText(frame, f"Raw:    {raw_sign}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            if stable_sign:
                sequence = sequencer.update(stable_sign)
                combo    = match_combo(sequence)

                cv2.putText(frame, f"SIGN: {stable_sign}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

                cv2.putText(frame, f"Seq: {' > '.join(sequence)}", (20, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

                if combo and combo is not active_effect:
                    active_effect = combo
                    effect_start  = time.monotonic()
                    palm_centres = [pts[9] for pts in hand_points]   # 9 = middle finger base
                    particles.burst(np.mean(palm_centres, axis=0), combo["colour"])
                    play_sound(*combo["sound"])

            else:
                cv2.putText(frame, "Holding...", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i, (name, state) in enumerate(zip(finger_names, fingers)):
                color = (0, 255, 0) if state == 1 else (0, 0, 255)
                cv2.putText(frame, f"{name}: {'UP' if state else 'DOWN'}",
                            (20, 130 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            # MediaPipe often drops the hand for a frame or two mid-transition
            missed_frames += 1
            if missed_frames == MISS_LIMIT:
                stabilizer.reset()
                sequencer.reset()
            cv2.putText(frame, "No Hand Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Visual effect overlay ---
        if active_effect:
            alpha = max(0.0, 1 - (time.monotonic() - effect_start) / EFFECT_SECONDS)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          active_effect["colour"], -1)
            cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)

            text = active_effect["name"]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] - text_size[1]) // 2 + 20
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

            if alpha == 0:
                active_effect = None
                sequencer.reset()

        now = time.monotonic()
        particles.update(now - last_tick)
        last_tick = now
        particles.draw(frame)

        cv2.imshow("Jutsu Code", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()