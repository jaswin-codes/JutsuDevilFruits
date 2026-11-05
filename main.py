import argparse
import math
import time
import cv2
import mediapipe as mp
import numpy as np
from hand_detector import HandDetector
from sign_model import SignModel
from stabilizer import SignStabilizer
from sequence import SequenceDetector
from combos import COMBOS, match_combo, MAX_COMBO_LENGTH
from effects import ParticleSystem, play_sound
from anime_effects import (draw_chidori, draw_rasengan, draw_clones, draw_fireball, face_aim,
                           fireball_path, CLONE_OFFSETS)
from game import Game
from battle import Battle
from battle_view import draw_battle

def main():
    parser = argparse.ArgumentParser(description="Jutsu Code - hand-sign jutsu recognition")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--game", action="store_true",
                      help="60-second challenge: perform the target jutsu for points (R restarts)")
    mode.add_argument("--battle", action="store_true",
                      help="AR monster battle: defeat waves of monsters with jutsu (R restarts)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    signs      = SignModel()                # real Naruto signs (both hands as one sign)
    detector   = HandDetector(max_hands=2)   # palm position for the effects
    stabilizer = SignStabilizer(buffer_size=10, confirm_threshold=7)
    sequencer  = SequenceDetector(max_length=MAX_COMBO_LENGTH)

    active_effect = None   # combo dict from combos.py while its effect plays
    effect_start  = 0.0
    # Max seconds per effect; hand-held ones also end once the hand is gone
    EFFECT_SECONDS = {"chidori": 8.0, "rasengan": 8.0, "clone": 3.0, "fireball": 2.5}
    HAND_EFFECTS = ("chidori", "rasengan")
    palm, palm_size = None, 40.0   # smoothed palm centre and hand size (wrist to middle knuckle)
    hand_spread = None     # distance between two palms (Rasengan size), None with one hand
    segmenter = None       # selfie segmentation, built on first Shadow Clone (slow to load)
    face_mesh, mouth = None, None   # built on first Fireball; mouth = last seen lip centre
    fire_scale, fire_dir = 1.0, 0.0   # mouth opening and head turn, locked at launch
    particles = ParticleSystem()
    game = Game(COMBOS, time.monotonic()) if args.game else None
    battle = None          # built on the first frame (needs the frame size)
    message, message_until = None, 0.0   # short battle notice, e.g. a cooldown
    last_tick = time.monotonic()
    MISS_LIMIT = 15        # frames with no hand or sign (~0.5s) before the combo is wiped
    missed_frames = 0

    print("Jutsu Code")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            break

        frame = cv2.flip(frame, 1)
        if args.battle and battle is None:
            battle = Battle(frame.shape[1], frame.shape[0])
        clean = frame.copy()   # without landmarks/HUD, for effects that copy the user
        raw_sign, sign_box = signs.detect(clean)
        frame, hands = detector.find_hands(frame)

        if hands:
            hand_points = [detector.get_landmark_list(h, frame.shape) for h in hands]
            # Palm = mean of wrist and knuckles; two hands -> between them (Rasengan)
            centres = [np.mean([pts[i] for i in (0, 5, 9, 13, 17)], axis=0) for pts in hand_points]
            palm_now = np.mean(centres, axis=0)
            palm_size = np.mean([np.hypot(*np.subtract(pts[0], pts[9])) for pts in hand_points])
            hand_spread = np.hypot(*np.subtract(*centres[:2])) if len(centres) == 2 else None
        elif sign_box:   # interlocked fingers often hide the hands from MediaPipe
            x1, y1, x2, y2 = sign_box
            palm_now, palm_size = np.array([(x1 + x2) / 2, (y1 + y2) / 2]), (x2 - x1) / 3
            hand_spread = None
        if hands or sign_box:
            palm = palm_now if palm is None else 0.5 * palm + 0.5 * palm_now   # damp jitter

        if sign_box:
            cv2.rectangle(frame, sign_box[:2], sign_box[2:], (0, 255, 100), 2)

        if hands or raw_sign:
            missed_frames = 0
            stable_sign = stabilizer.update(raw_sign)

            cv2.putText(frame, f"Raw:    {raw_sign}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            if stable_sign:
                sequence = sequencer.update(stable_sign)
                combo    = match_combo(sequence)

                cv2.putText(frame, f"SIGN: {stable_sign}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

                cv2.putText(frame, f"Seq: {' > '.join(sequence)}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

                if combo and combo is not active_effect and battle and not battle.on_combo(combo["effect"]):
                    # Cooling down: drop the combo but keep last_sign, so a held sign can't re-fire every frame
                    sequencer.sequence.clear()
                    if not battle.over():
                        message = f"{combo['name']} - cooling down {math.ceil(battle.cooldown_left(combo['effect']))}s"
                        message_until = time.monotonic() + 1.5
                elif combo and combo is not active_effect:
                    active_effect = combo
                    effect_start  = time.monotonic()
                    particles.burst(palm, combo["colour"])
                    if combo["effect"] == "clone":   # smoke puff where each clone appears
                        for offset in CLONE_OFFSETS:
                            particles.burst((palm[0] + offset * frame.shape[1], palm[1]),
                                            (170, 170, 170), count=120, speed=350)
                    play_sound(*combo["sound"])
                    if game:
                        game.on_combo(combo, time.monotonic())

            else:
                cv2.putText(frame, "Holding...", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
        else:
            # Detection often drops for a frame or two mid-transition
            missed_frames += 1
            stabilizer.update(None)
            if missed_frames == MISS_LIMIT:
                stabilizer.reset()
                sequencer.reset()
            cv2.putText(frame, "No Hand Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Visual effect overlay ---
        now = time.monotonic()
        dt = now - last_tick
        killed = []            # monster positions defeated this frame (battle mode)
        if active_effect:
            t = now - effect_start
            kind = active_effect["effect"]
            if kind == "chidori":
                radius = palm_size * 1.1 * (1 + 0.8 * min(t / 3, 1))   # charges up
                draw_chidori(frame, palm, radius, t)
                if battle:
                    killed = battle.melee(palm, radius, dt)
            elif kind == "rasengan":   # pull your hands apart to grow it
                size = palm_size * 1.2 if hand_spread is None else np.clip(
                    hand_spread * 0.45, palm_size * 0.6, palm_size * 3)
                draw_rasengan(frame, palm, size, t)
                if battle:   # bigger Rasengan hits harder
                    killed = battle.melee(palm, size, dt, power=size / (palm_size * 1.2))
            elif kind == "clone":
                if segmenter is None:
                    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                mask = segmenter.process(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).segmentation_mask
                draw_clones(frame, clean, mask, t, EFFECT_SECONDS[kind])
            elif kind == "fireball":
                if face_mesh is None:
                    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
                if t < 0.5:   # aim while it gathers; once launched it keeps that course
                    faces = face_mesh.process(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).multi_face_landmarks
                    if faces:
                        mouth, fire_scale, fire_dir = face_aim([(p.x, p.y) for p in faces[0].landmark])
                        mouth = mouth * (frame.shape[1], frame.shape[0])
                origin = mouth if mouth is not None else palm
                draw_fireball(frame, origin, t, EFFECT_SECONDS[kind], fire_scale, fire_dir)
                if battle and t >= 0.5:   # only the launched fireball hits
                    centre, radius = fireball_path(origin, t, EFFECT_SECONDS[kind], fire_scale, fire_dir,
                                                   frame.shape)
                    killed = battle.fireball(origin[0], centre[0], radius, fire_dir)

            if t < 2.0:   # jutsu name banner near the top
                text = active_effect["name"]
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
                cv2.putText(frame, text, ((frame.shape[1] - text_size[0]) // 2, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

            if t > EFFECT_SECONDS[kind] or (kind in HAND_EFFECTS and missed_frames >= MISS_LIMIT):
                active_effect = None
                sequencer.reset()

        if battle:
            battle.update(dt)
            for pos in killed:
                particles.burst(pos, (180, 60, 160), count=150, speed=400)
            draw_battle(frame, battle, message if now < message_until else None)

        particles.update(dt)
        last_tick = now
        particles.draw(frame)

        if game:
            h = frame.shape[0]
            if game.over(now):
                hud = [f"TIME UP - SCORE {game.score}", "Press R to play again"]
            else:
                hud = [f"TARGET: {COMBOS[game.target]['name']}",
                       f"Signs: {' > '.join(game.target)}",
                       f"Score: {game.score}   Time: {game.time_left(now):.0f}s"]
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (20, h - 20 - 30 * (len(hud) - 1 - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Jutsu Code", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and game:
            game = Game(COMBOS, time.monotonic())
        if key == ord('r') and battle:
            battle = Battle(frame.shape[1], frame.shape[0])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()