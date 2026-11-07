# JutsuDevilFruits

Anime powers over your webcam: Naruto jutsu from real hand signs and One Piece devil-fruit
abilities from hand gestures, with live visual effects, built with Python, MediaPipe, and OpenCV.

## How it started

This project began in July 2026 as **Jutsu Code** (v1, the `jutsu_code_stage1` repository): a
webcam app that recognised Naruto hand signs with hand-written finger rules and fired a jutsu
when you made a combo. It has grown since: a trained model now reads the real hand signs, the
jutsu became anime-style effects that follow your hands and body, and challenge, monster battle
and two-player duel modes were added. Then it left the Naruto world: a One Piece mode gives you
Luffy's Gomu Gomu devil fruit (rubber stretching, Pistol, Gatling, Gear Second and Gear Fourth),
then Mera Mera and Goro Goro, and a Bleach world lets you draw a sword out of thin air and transform.
With more than jutsu in it now, it was renamed **JutsuDevilFruits**.

## Features

- Real Naruto hand signs (12 zodiac signs + prayer hands) from a trained YOLOX-Nano model
  ([Kazuhito00/NARUTO-HandSignDetection](https://github.com/Kazuhito00/NARUTO-HandSignDetection), MIT)
- Live webcam hand tracking (MediaPipe) to place the effects on your palms
- Temporal stabilization to reduce flicker
- Sequence detection for combo matching
- Anime-style jutsu effects: Chidori lightning and a spinning Rasengan that follow your palm,
  Shadow Clones of you (selfie segmentation), invisibility against a learned background, a Fireball from your mouth (face mesh),
  a summoned creature rising from smoke, and Sharingan eyes drawn on your irises,
  plus particle bursts and sound (sound on Windows)
- Body control: hand spread sizes the Rasengan, Chidori charges while held, and the Fireball
  is sized by how wide you open your mouth and aimed by turning your head
- Sign chart: a photo of every hand sign with its name (0 in the Naruto menu, H in any Naruto mode)
- Game mode: 60-second challenge with a target jutsu and score, with photos of its signs
- Battle mode: AR monster battle in your webcam image, with waves, HP, score and jutsu cooldowns
- Duel mode: two players side by side in front of one webcam, split screen, fight each other with jutsu
- One Piece mode: Luffy's rubber body. Make a fist and it shoots out on a stretched arm
  (Gomu Gomu no Pistol), pump it for a Gatling burst of fists; pinch your mouth or cheek and pull, and
  the skin stretches out and snaps back. Hold a V sign for Gear Second (pink, steaming, faster
  Pistol) or four fingers for Gear Fourth (puffed-up arms, chest and fists)
- Mera Mera (fire fruit): a fist held up burns, flames rising off it. Punch the fist at the
  camera and a Hiken fire blast rushes at you
- Goro Goro (lightning fruit): point one finger up and lightning crackles around your hand (Vari).
  Raise both open hands and hold them up for El Thor, a bolt from the top of the screen
- Bleach: hold a fist low and pull it up fast to draw a glowing sword; it follows your fist's
  angle. Swing it for Getsuga Tensho (a flying crescent slash). Grip the sword wrist with your
  other hand for a second for Bankai (dark room, red flames, black blade). Drag an open hand
  down over your face for a Hollow mask (bigger slashes for 15 s)

## Current Sign Set

Ne, Ushi, Tora, U, Tatsu, Mi, Uma, Hitsuji, Saru, Tori, Inu, I (the 12 zodiac signs) and
Gassho (prayer hands). See JUTSU_GUIDE.md. The photos in the sign chart are in assets/signs
(cropped from the model author's README, MIT; source and licence in assets/signs/SOURCE.md).

## Current Combos

Sign orders follow the anime:

- Mi -> Tora -> Saru -> I -> Uma -> Tora: KATON - FIREBALL JUTSU
- Hitsuji -> Mi -> Tora: SHADOW CLONE JUTSU
- Ushi -> U -> Saru: CHIDORI
- Gassho: RASENGAN (no canon signs)
- Ne -> Inu -> Tori: HIDING WITH CAMOUFLAGE (invisibility; our own sign pick)
- Tatsu -> Ne: SUBSTITUTION JUTSU (the next hit lands on a log; our own sign pick)
- I -> Inu -> Tori -> Saru -> Hitsuji, then slap a palm down: SUMMONING JUTSU
- No signs: cover one eye with a hand, then lower it: SHARINGAN

Add a combo of any length with one line in combos.py.

## Project Structure

- main.py: App entry point and render loop
- sign_model.py: Runs the hand-sign model (model/yolox_nano_with_post.onnx, MIT, see model/LICENSE)
- hand_detector.py: MediaPipe hand landmarks, used to place effects on the palm
- stabilizer.py: Temporal smoothing for stable sign confirmation
- sequence.py: Sign change sequence tracking
- combos.py: Combo definitions (name, colour, sound) and lookup
- effects.py: Particle bursts and generated jutsu sounds
- anime_effects.py: Chidori, Rasengan, Shadow Clone, Fireball, summon and Sharingan drawing (numpy + OpenCV)
- gestures.py: Non-sign gestures: the summoning palm slap and the Sharingan eye cover
- game.py: Timed challenge mode
- sign_chart.py: The hand-sign photo chart and the challenge's row of sign photos
- assets/signs/: One photo per sign (source and licence in SOURCE.md)
- battle.py: Monster battle logic (waves, hits, HP, cooldowns), no camera or drawing
- battle_view.py: Draws the monsters and battle HUD
- duel.py: Two-player duel rules (HP, cooldowns, hits, blocks), no camera or drawing
- duel_view.py: Draws the duel HUD (HP bars, cooldowns, winner screen)
- one_piece.py: One Piece mode (rubber stretch and bulge warps, fist/pinch/gear-sign gestures,
  snap-back spring, Gatling pump detection, Gear Second/Fourth, its loop)
- mera.py: Mera Mera mode (fist flames, Hiken punch detection from knuckle growth, its loop)
- goro.py: Goro Goro mode (pointing and raised-hands gestures, Vari and El Thor, its loop)
- bleach.py: Bleach mode (sword draw, swing, wrist grip and face swipe detection; sword, Getsuga,
  Bankai aura and Hollow mask drawing; its loop)
- test_logic.py, test_battle.py, test_duel.py, test_one_piece.py, test_mera.py,
  test_goro.py, test_bleach.py, test_sign_chart.py, test_menu.py (fake-camera menu run): Webcam-free tests
- JUTSU_GUIDE.md: End-user guide for sign execution
- PROJECT_OVERVIEW.md: Detailed build notes and stage progression

## Requirements

- Python 3.11
- Webcam

Python packages:

- mediapipe 0.10.21 (installs OpenCV as opencv-contrib-python)
- onnxruntime (runs the hand-sign model on CPU)
- pytest (tests only)

## Quick Start (Windows PowerShell)

1. Create and activate a Python 3.11 virtual environment (mediapipe 0.10.21 doesn't support 3.13+):

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python main.py
```

   A start menu over the camera picks the world: 1 Naruto, 2 One Piece, 3 Bleach. Naruto then picks the
   mode: 1 free play, 2 challenge, 3 monster battle, 4 duel, or 0 for the hand-sign chart.
   One Piece picks the devil fruit: 1 Gomu Gomu, 2 Mera Mera, 3 Goro Goro.
   Press M to go back a menu. In any Naruto mode, H opens the sign chart and H closes it;
   the game clock, monsters and duel countdown wait while it is open.
   Use another webcam with `python main.py --camera 1`.
   Skip the menu with a flag: `--free`, `--game` (R restarts when time is up),
   `--battle` (R restarts after defeat) or `--duel` (left half = player 1, right half = player 2,
   best of 3; R starts a rematch), `--one-piece` (Gomu Gomu), `--mera`, `--goro` or `--bleach`.

4. Quit by pressing Q in the OpenCV window.

## Running Tests

The recognition, battle and duel logic are tested without a webcam:

```powershell
python -m pytest
```

GitHub Actions runs the same tests on every push and pull request (`.github/workflows/tests.yml`).

## How Detection Works

1. Capture frame from webcam.
2. Run the sign model on the frame: it returns the most confident sign and its box.
3. Track hand landmarks so effects follow your palm.
4. Ignore signs below 0.6 confidence.
5. Stabilize sign over recent frames.
6. Append only sign changes to a short sequence buffer.
7. Match sequence against combo table and trigger effect.

## Troubleshooting

- No camera feed:
  - Run `python main.py --camera 1` (or 2, 3...) to pick another webcam.
- Signs flicker:
  - Improve lighting, reduce motion, and keep both hands in the middle of the frame.
- Combo not triggering:
  - Hold each sign long enough to stabilize before changing.
- Startup import errors:
  - Ensure the virtual environment is active and dependencies are installed.
