# Jutsu Code

Real-time hand-sign recognition with jutsu combo detection and visual effects, built with Python, MediaPipe, and OpenCV.

## Features

- Real Naruto hand signs (12 zodiac signs + prayer hands) from a trained YOLOX-Nano model
  ([Kazuhito00/NARUTO-HandSignDetection](https://github.com/Kazuhito00/NARUTO-HandSignDetection), MIT)
- Live webcam hand tracking (MediaPipe) to place the effects on your palms
- Temporal stabilization to reduce flicker
- Sequence detection for combo matching
- Anime-style jutsu effects: Chidori lightning and a spinning Rasengan that follow your palm,
  Shadow Clones of you (selfie segmentation), invisibility against a learned background, and a Fireball from your mouth (face mesh),
  plus particle bursts and sound (sound on Windows)
- Body control: hand spread sizes the Rasengan, Chidori charges while held, and the Fireball
  is sized by how wide you open your mouth and aimed by turning your head
- Game mode: 60-second challenge with a target jutsu and score
- Battle mode: AR monster battle in your webcam image, with waves, HP, score and jutsu cooldowns
- Duel mode: two players side by side in front of one webcam, split screen, fight each other with jutsu
- One Piece mode: Luffy's rubber body. Make a fist and it shoots out on a stretched arm
  (Gomu Gomu no Pistol); pinch your mouth or cheek and pull, and the skin stretches out and snaps back

## Current Sign Set

Ne, Ushi, Tora, U, Tatsu, Mi, Uma, Hitsuji, Saru, Tori, Inu, I (the 12 zodiac signs) and
Gassho (prayer hands). See JUTSU_GUIDE.md.

## Current Combos

Sign orders follow the anime:

- Mi -> Tora -> Saru -> I -> Uma -> Tora: KATON - FIREBALL JUTSU
- Hitsuji -> Mi -> Tora: SHADOW CLONE JUTSU
- Ushi -> U -> Saru: CHIDORI
- Gassho: RASENGAN (no canon signs)
- Ne -> Inu -> Tori: HIDING WITH CAMOUFLAGE (invisibility; our own sign pick)
- Tatsu -> Ne: SUBSTITUTION JUTSU (the next hit lands on a log; our own sign pick)

Add a combo of any length with one line in combos.py.

## Project Structure

- main.py: App entry point and render loop
- sign_model.py: Runs the hand-sign model (model/yolox_nano_with_post.onnx, MIT, see model/LICENSE)
- hand_detector.py: MediaPipe hand landmarks, used to place effects on the palm
- stabilizer.py: Temporal smoothing for stable sign confirmation
- sequence.py: Sign change sequence tracking
- combos.py: Combo definitions (name, colour, sound) and lookup
- effects.py: Particle bursts and generated jutsu sounds
- anime_effects.py: Chidori, Rasengan, Shadow Clone and Fireball drawing (numpy + OpenCV)
- game.py: Timed challenge mode
- battle.py: Monster battle logic (waves, hits, HP, cooldowns), no camera or drawing
- battle_view.py: Draws the monsters and battle HUD
- duel.py: Two-player duel rules (HP, cooldowns, hits, blocks), no camera or drawing
- duel_view.py: Draws the duel HUD (HP bars, cooldowns, winner screen)
- one_piece.py: One Piece mode (rubber stretch warp, fist/pinch gestures, snap-back spring, its loop)
- test_logic.py, test_battle.py, test_duel.py, test_one_piece.py: Webcam-free tests
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

   A start menu over the camera picks the world: 1 Naruto, 2 One Piece. Naruto then picks the
   mode: 1 free play, 2 challenge, 3 monster battle, 4 duel. Press M to go back a menu.
   Use another webcam with `python main.py --camera 1`.
   Skip the menu with a flag: `--free`, `--game` (R restarts when time is up),
   `--battle` (R restarts after defeat) or `--duel` (left half = player 1, right half = player 2,
   best of 3; R starts a rematch) or `--one-piece`.

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
