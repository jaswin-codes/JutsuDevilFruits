# Jutsu Code

Real-time hand-sign recognition with jutsu combo detection and visual effects, built with Python, MediaPipe, and OpenCV.

## Features

- Live webcam hand tracking (21 MediaPipe landmarks)
- Finger-state extraction for 5 fingers
- Rule-based sign recognition (no model training)
- Temporal stabilization to reduce flicker
- Sequence detection for combo matching
- Two-hand signs (e.g. both hands open = `Ushi+Ushi`)
- Anime-style jutsu effects: Chidori lightning and a spinning Rasengan that follow your palm,
  Shadow Clones of you (selfie segmentation), and a Fireball from your mouth (face mesh),
  plus particle bursts and sound (sound on Windows)
- Game mode: 60-second challenge with a target jutsu and score

## Current Sign Set

- Tora = index + middle up
- Ushi = open hand
- Ne = index only
- Inu = fist
- Tori = index + pinky up

## Current Combos

- Ne -> Tora -> Inu: KATON - FIREBALL JUTSU
- Ushi -> Ne -> Tora: SHADOW CLONE JUTSU
- Tori -> Inu -> Ushi: CHIDORI
- Ushi+Ushi -> Tora+Tora (both hands): RASENGAN

Two-hand signs are written `Left+Right`, meaning your own left and right hand.
Add a combo of any length with one line in combos.py.

## Project Structure

- main.py: App entry point and render loop
- hand_detector.py: Hand landmark detection + finger state extraction
- signs.py: Sign definitions and matching
- stabilizer.py: Temporal smoothing for stable sign confirmation
- sequence.py: Sign change sequence tracking
- combos.py: Combo definitions (name, colour, sound) and lookup
- effects.py: Particle bursts and generated jutsu sounds
- anime_effects.py: Chidori, Rasengan, Shadow Clone and Fireball drawing (numpy + OpenCV)
- game.py: Timed challenge mode
- test_logic.py: Webcam-free tests
- JUTSU_GUIDE.md: End-user guide for sign execution
- PROJECT_OVERVIEW.md: Detailed build notes and stage progression

## Requirements

- Python 3.11
- Webcam

Python packages:

- mediapipe 0.10.21 (installs OpenCV as opencv-contrib-python)
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

   Use another webcam with `python main.py --camera 1`.
   Play the challenge with `python main.py --game` (R restarts when time is up).

4. Quit by pressing Q in the OpenCV window.

## Running Tests

The recognition logic is tested without a webcam:

```powershell
python -m pytest
```

## How Detection Works

1. Capture frame from webcam.
2. Detect hand landmarks.
3. Convert landmarks to [thumb, index, middle, ring, pinky] states.
4. Match finger states to a known sign pattern.
5. Stabilize sign over recent frames.
6. Append only sign changes to a short sequence buffer.
7. Match sequence against combo table and trigger effect.

## Troubleshooting

- No camera feed:
  - Run `python main.py --camera 1` (or 2, 3...) to pick another webcam.
- Signs flicker:
  - Improve lighting, reduce motion, and keep hand centered.
- Combo not triggering:
  - Hold each sign long enough to stabilize before changing.
- Startup import errors:
  - Ensure the virtual environment is active and dependencies are installed.
