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
                           fireball_path, draw_invisible, draw_log, BackgroundPlate, CLONE_OFFSETS)
from game import Game
from battle import Battle
from battle_view import draw_battle
from duel import Duel
from duel_view import draw_duel
import one_piece
from sign_chart import show_chart, draw_sign_strip

# Max seconds per effect; hand-held ones also end once the hand is gone
EFFECT_SECONDS = {"chidori": 8.0, "rasengan": 8.0, "clone": 3.0, "fireball": 2.5, "invisible": 8.0,
                  "substitution": 1.5}
HAND_EFFECTS = ("chidori", "rasengan")
MISS_LIMIT = 15        # frames with no hand or sign (~0.5s) before the combo is wiped
PALM_POINTS = (0, 5, 9, 13, 17)   # wrist and knuckles


class Player:
    """One person's sign reading and jutsu. Plain/--game/--battle have one player owning the
    whole frame; --duel has two, each owning one screen half (columns x0..x1)."""

    def __init__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        self.stabilizer = SignStabilizer(buffer_size=10, confirm_threshold=7)
        self.sequencer = SequenceDetector(max_length=MAX_COMBO_LENGTH)
        self.active_effect = None   # combo dict from combos.py while its effect plays
        self.effect_start = 0.0
        self.palm, self.palm_size = None, 40.0   # smoothed palm centre and hand size (wrist to middle knuckle)
        self.hand_spread = None     # distance between two palms (Rasengan size), None with one hand
        self.missed_frames = 0
        self.mouth = None           # last seen lip centre (Fireball)
        self.fire_scale, self.fire_dir = 1.0, 0.0   # mouth opening and head turn, locked at launch
        self.message, self.message_until = None, 0.0   # short notice, e.g. a cooldown


MODES = [("free", "FREE PLAY", "every jutsu, no rules"),
         ("game", "CHALLENGE", "60 s, perform the target jutsu"),
         ("battle", "MONSTER BATTLE", "survive waves of monsters"),
         ("duel", "DUEL", "2 players side by side, best of 3")]
CHART_ITEM = ("chart", "HAND SIGNS", "learn every sign from photos (H in a game)")
WORLDS = [("naruto", "NARUTO", "hand-sign jutsu: free play, challenge, battle, duel"),
          ("one_piece", "ONE PIECE", "Gomu Gomu: stretch your fist and your face")]


def main():
    parser = argparse.ArgumentParser(description="Jutsu Code - hand-sign jutsu recognition. "
                                                 "Without a mode flag a start menu picks the mode.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--free", action="store_true", help="free play: every jutsu, no rules")
    mode.add_argument("--game", action="store_true",
                      help="60-second challenge: perform the target jutsu for points (R restarts)")
    mode.add_argument("--battle", action="store_true",
                      help="AR monster battle: defeat waves of monsters with jutsu (R restarts)")
    mode.add_argument("--duel", action="store_true",
                      help="two players side by side, split screen: fight each other with jutsu (R restarts)")
    mode.add_argument("--one-piece", action="store_true",
                      help="One Piece: Luffy's rubber stretching (fist = Pistol, pump it = Gatling, pinch and pull, "
                           "V sign = Gear Second, four fingers = Gear Fourth)")
    args = parser.parse_args()
    mode = next((m for m in [m for m, *_ in MODES] + ["one_piece"] if getattr(args, m)), None)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    signs = SignModel()   # real Naruto signs (both hands as one sign); loaded once for every mode

    print("Jutsu Code")
    print("Press Q to quit, M for the menu.")
    while True:
        if not mode:
            world = menu(cap, "JUTSU CODE", WORLDS)
            mode = menu(cap, "NARUTO", MODES, CHART_ITEM) if world == "naruto" else world
            while mode == "chart":   # back to the Naruto menu when the chart closes
                mode = None if show_chart(cap) == "quit" else menu(cap, "NARUTO", MODES, CHART_ITEM)
            if mode is None:
                break
            if not mode:
                continue  # M in the Naruto menu: back to the worlds
        if (one_piece.play(cap) if mode == "one_piece" else play(cap, signs, mode)) == "quit":
            break
        mode = None       # M pressed: back to the menu

    cap.release()
    cv2.destroyAllWindows()


def menu(cap, title, items, zero_item=None):
    """Menu over the live camera image. Returns the picked item key, "" for M (back), None to quit.
    zero_item: an extra (key, name, about) row picked with 0."""
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            return None
        frame = (cv2.flip(frame, 1) * 0.35).astype(np.uint8)   # dimmed, so the text stands out
        w = frame.shape[1]
        size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
        cv2.putText(frame, title, ((w - size[0]) // 2, 90), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 165, 255), 3)
        rows = [(k + 1, *item[1:]) for k, item in enumerate(items)] + ([(0, *zero_item[1:])] if zero_item else [])
        step = 70 if len(rows) <= 4 else 58
        for k, (number, name, about) in enumerate(rows):
            y = 155 + step * k
            cv2.putText(frame, f"{number}  {name}", (110, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, about, (150, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)
        first = 0 if zero_item else 1
        cv2.putText(frame, f"Press {first}-{len(items)} to pick, Q to quit, M goes back", (60, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow("Jutsu Code", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        if key == ord('m'):
            return ""
        if ord('1') <= key < ord('1') + len(items):
            return items[key - ord('1')][0]
        if zero_item and key == ord('0'):
            return zero_item[0]


def play(cap, signs, mode):
    """Runs one mode until Q ("quit") or M ("menu")."""
    duel_mode = mode == "duel"
    detector = HandDetector(max_hands=4 if duel_mode else 2)   # palm position for the effects
    ui = 0.7 if duel_mode else 1.0                          # text scale: a duel half is 320 px wide

    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    plate = BackgroundPlate()   # empty scene for invisibility, learned from the frames
    frame_no = 0
    face_mesh = None            # built on first Fireball
    particles = ParticleSystem()
    game = Game(COMBOS, time.monotonic()) if mode == "game" else None
    players = battle = duel = None   # built on the first frame (need the frame size)
    last_tick = time.monotonic()

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            return "quit"

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if players is None:
            players = [Player(0, w // 2), Player(w // 2, w)] if duel_mode else [Player(0, w)]
        if mode == "battle" and battle is None:
            battle = Battle(w, h)
        if duel_mode and duel is None:
            duel = Duel(w, h)
        clean = frame.copy()   # without landmarks/HUD, for effects that copy the user
        detections = signs.detect_halves(clean) if duel_mode else [signs.detect(clean)]
        frame, hands = detector.find_hands(frame)
        hand_points = [detector.get_landmark_list(hd, frame.shape) for hd in hands]
        hand_centres = [np.mean([pts[k] for k in PALM_POINTS], axis=0) for pts in hand_points]

        for i, (p, (raw_sign, sign_box)) in enumerate(zip(players, detections)):
            arena = battle or (duel.fighters[i] if duel else None)   # gates combos by cooldown
            tx = p.x0 + 20
            # Hands belong to the player whose half their palm is in
            owned = [k for k, c in enumerate(hand_centres) if not duel_mode or (c[0] >= w / 2) == (i == 1)]
            centres = [hand_centres[k] for k in owned]
            mine = [hand_points[k] for k in owned]

            if centres:
                # Palm = mean of wrist and knuckles; two hands -> between them (Rasengan)
                palm_now = np.mean(centres, axis=0)
                p.palm_size = np.mean([np.hypot(*np.subtract(pts[0], pts[9])) for pts in mine])
                p.hand_spread = np.hypot(*np.subtract(*centres[:2])) if len(centres) == 2 else None
            elif sign_box:   # interlocked fingers often hide the hands from MediaPipe
                x1, y1, x2, y2 = sign_box
                palm_now, p.palm_size = np.array([(x1 + x2) / 2, (y1 + y2) / 2]), (x2 - x1) / 3
                p.hand_spread = None
            if centres or sign_box:
                p.palm = palm_now if p.palm is None else 0.5 * p.palm + 0.5 * palm_now   # damp jitter

            if sign_box:
                cv2.rectangle(frame, sign_box[:2], sign_box[2:], (0, 255, 100), 2)

            if centres or raw_sign:
                p.missed_frames = 0
                stable_sign = p.stabilizer.update(raw_sign)

                cv2.putText(frame, f"Raw:    {raw_sign}", (tx, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (150, 150, 150), 1)

                if stable_sign:
                    sequence = p.sequencer.update(stable_sign)
                    combo    = match_combo(sequence)

                    cv2.putText(frame, f"SIGN: {stable_sign}", (tx, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * ui, (0, 255, 100), 3)

                    cv2.putText(frame, f"Seq: {' > '.join(sequence)}", (tx, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ui, (200, 200, 0), 2)

                    if combo and combo is not p.active_effect and arena and not arena.on_combo(combo["effect"]):
                        # Cooling down: drop the combo but keep last_sign, so a held sign can't re-fire every frame
                        p.sequencer.sequence.clear()
                        if not arena.over():
                            p.message = f"{combo['name']} - cooling down {math.ceil(arena.cooldown_left(combo['effect']))}s"
                            p.message_until = time.monotonic() + 1.5
                    elif combo and combo is not p.active_effect:
                        p.active_effect = combo
                        p.effect_start  = time.monotonic()
                        particles.burst(p.palm, combo["colour"])
                        if combo["effect"] == "clone":   # smoke puff where each clone appears
                            for offset in CLONE_OFFSETS:
                                particles.burst((p.palm[0] + offset * (p.x1 - p.x0), p.palm[1]),
                                                (170, 170, 170), count=120, speed=350)
                        play_sound(*combo["sound"])
                        if game:
                            game.on_combo(combo, time.monotonic())

                else:
                    cv2.putText(frame, "Holding...", (tx, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * ui, (0, 180, 255), 2)
            else:
                # Detection often drops for a frame or two mid-transition
                p.missed_frames += 1
                p.stabilizer.update(None)
                if p.missed_frames == MISS_LIMIT:
                    p.stabilizer.reset()
                    p.sequencer.reset()
                cv2.putText(frame, "No Hand Detected", (tx, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui, (0, 0, 255), 2)

        # --- Visual effect overlay ---
        now = time.monotonic()
        dt = now - last_tick
        killed = []            # monster positions defeated this frame (battle mode)
        frame_no += 1
        kinds = [p.active_effect["effect"] if p.active_effect else None for p in players]
        mask = None            # selfie mask (1 = person): for clones, invisibility, background learning
        if "clone" in kinds or "invisible" in kinds or frame_no % 10 == 0:
            mask = segmenter.process(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).segmentation_mask
            plate.update(clean, mask)

        aiming = [p for p, kind in zip(players, kinds) if kind == "fireball" and now - p.effect_start < 0.5]
        if aiming:   # aim while it gathers; once launched it keeps that course
            if face_mesh is None:
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=len(players))
            for face in face_mesh.process(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).multi_face_landmarks or []:
                mouth, scale, direction = face_aim([(lm.x, lm.y) for lm in face.landmark])
                mouth = mouth * (w, h)
                owner = players[int(mouth[0] >= w / 2) if duel_mode else 0]
                if owner in aiming:
                    owner.mouth, owner.fire_scale, owner.fire_dir = mouth, scale, direction

        for i, (p, kind) in enumerate(zip(players, kinds)):
            if kind is None:
                continue
            t = now - p.effect_start
            if kind == "chidori":
                radius = p.palm_size * 1.1 * (1 + 0.8 * min(t / 3, 1))   # charges up
                draw_chidori(frame, p.palm, radius, t)
                if battle:
                    killed += battle.melee(p.palm, radius, dt)
                if duel:
                    duel.melee(i, p.palm, radius, dt)
            elif kind == "rasengan":   # pull your hands apart to grow it
                size = p.palm_size * 1.2 if p.hand_spread is None else np.clip(
                    p.hand_spread * 0.45, p.palm_size * 0.6, p.palm_size * 3)
                draw_rasengan(frame, p.palm, size, t)
                power = size / (p.palm_size * 1.2)   # bigger Rasengan hits harder
                if battle:
                    killed += battle.melee(p.palm, size, dt, power=power)
                if duel:
                    duel.melee(i, p.palm, size, dt, power=power)
            elif kind == "invisible":
                own = np.zeros_like(mask)            # only the caster vanishes, not the opponent
                own[:, p.x0:p.x1] = mask[:, p.x0:p.x1]
                draw_invisible(frame, own, plate, t, EFFECT_SECONDS[kind])
            elif kind == "clone":   # drawn into the caster's columns only (views write through)
                draw_clones(frame[:, p.x0:p.x1], clean[:, p.x0:p.x1], mask[:, p.x0:p.x1], t, EFFECT_SECONDS[kind])
            elif kind == "substitution":
                if not (battle or duel):   # free play: swap with a log on the spot; in a fight it waits for a hit
                    draw_log(frame, p.palm, p.palm_size * 5, t, EFFECT_SECONDS[kind])
            elif kind == "fireball":
                origin = p.mouth if p.mouth is not None else p.palm
                direction = p.fire_dir
                if duel and direction * duel.toward(i) >= 0:   # facing the camera: straight at the opponent
                    direction = duel.toward(i)
                draw_fireball(frame, origin, t, EFFECT_SECONDS[kind], p.fire_scale, direction)
                if (battle or duel) and t >= 0.5:   # only the launched fireball hits
                    centre, radius = fireball_path(origin, t, EFFECT_SECONDS[kind], p.fire_scale, direction,
                                                   frame.shape)
                    if battle:
                        killed += battle.fireball(origin[0], centre[0], radius, direction)
                    else:
                        duel.fireball(i, centre[0], radius)

            if t < 2.0:   # jutsu name banner near the top (in a duel: below that player's sign readout)
                text = p.active_effect["name"]
                scale = 1.2
                text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0]
                scale = min(scale, scale * (p.x1 - p.x0 - 20) / text_w)
                text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0]
                cv2.putText(frame, text, ((p.x0 + p.x1 - text_w) // 2, 165 if duel else 60),
                            cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 3 if scale > 0.8 else 2)

            if t > EFFECT_SECONDS[kind] or (kind in HAND_EFFECTS and p.missed_frames >= MISS_LIMIT):
                p.active_effect = None
                p.sequencer.reset()

        messages = [p.message if now < p.message_until else None for p in players]
        if battle:
            battle.update(dt)
            for pos in killed:
                particles.burst(pos, (180, 60, 160), count=150, speed=400)
            draw_battle(frame, battle, messages[0])
        if duel:
            duel.update(dt)
            draw_duel(frame, duel, messages)

        particles.update(dt)
        last_tick = now
        particles.draw(frame)

        if game:
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
            return "quit"
        if key == ord('m'):
            return "menu"
        if key == ord('h'):
            paused = show_chart(cap, "H goes back to the game", close_keys=(ord('h'),))
            if paused == "quit":
                return "quit"
            # The world stood still: shift absolute clocks, and dt restarts from now (duel countdown, monsters)
            if game:
                game.pause(paused)
            for p in players:
                p.effect_start += paused
                p.message_until += paused
                p.stabilizer.reset()
                p.sequencer.reset()
            last_tick = time.monotonic()
        if key == ord('r') and game:
            game = Game(COMBOS, time.monotonic())
        if key == ord('r') and battle:
            battle = Battle(w, h)
        if key == ord('r') and duel:
            duel = Duel(w, h)

if __name__ == "__main__":
    main()
