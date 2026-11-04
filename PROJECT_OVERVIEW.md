# Jutsu Code - Project Overview

> A real-time hand gesture recognition system inspired by Naruto jutsu hand signs,
> Built using computer vision and gesture sequence detection. 

## Tools Used
Built using Python, MediaPipe, and OpenCV.
AI coding assistant Claude was used during development for code generation and debugging support
but architecture, logic, testing and everything else was designed by myself

---

## Project Summary
**Jutsu Code** is a Python-based computer vision project that uses a webcam to detect
hand gestures in real time, recognise sequences of gestures, and trigger visual effects
when a predefined jutsu combo is performed. Built from scratch without any pre-trained
gesture classification models - all recognition logic is rule-based and custom written.

---

## Technologies & Libraries Used

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| MediaPipe | Hand landmark detection (21-point skeleton) |
| OpenCV (cv2) | Camera input, frame processing, visual overlay |
| Python `collections.deque` | Frame buffer for stabilization and sequence tracking |

---

## Development Stages

### Stage 1 - Hand Detection
Set up the webcam pipeline using OpenCV. Integrated MediaPipe's hand tracking solution
to detect a hand in each frame and extract 21 landmark points representing joints and
fingertips. Resolved a Python 3.13 / MediaPipe compatibility issue by migrating to a
Python 3.11 virtual environment.

**Key concepts:** Camera feed capture, BGR→RGB conversion, MediaPipe landmark extraction,
virtual environments, dependency management.

---

### Stage 2 - Finger State Detection
Converted the 21 raw landmark coordinates into a binary finger state array:
`[thumb, index, middle, ring, pinky]` where `1 = up` and `0 = down`.

Used y-coordinate comparison (fingertip vs lower joint) for the four fingers, and
x-coordinate comparison relative to the palm anchor for the thumb to account for
mirrored camera orientation.

**Key concepts:** Coordinate geometry, landmark indexing, mirrored camera logic,
modular class design.

---

### Stage 3 - Sign Definition & Matching
Defined 5 named hand signs as finger state patterns. Built a `match_sign()` function
supporting wildcard (`-1`) values to ignore unreliable fingers (e.g. thumb) per sign.

**Signs defined:** Tora, Ushi, Ne, Inu, Tori
**Key concepts:** Pattern matching, wildcard logic, separation of data from logic.

---

### Stage 4 - Stabilization
Implemented a rolling frame buffer (`deque`) to prevent flickering. A sign is only
confirmed after appearing consistently across a threshold number of recent frames
(7 out of 10 by default, ~0.33 seconds at 30fps).

**Key concepts:** Temporal smoothing, deque buffers, confidence thresholds,
signal stabilization.

---

### Stage 5 - Sign Change Detection
Added logic to only record a sign into the sequence when it changes from the previous
confirmed sign. Prevents duplicate entries from a held gesture flooding the sequence buffer.

**Key concepts:** State tracking, change detection, deduplication.

---

### Stage 6 - Sequence Detection & Combo Matching
Built a sequence buffer storing the last N confirmed sign changes. Defined jutsu combos
as ordered tuples of sign names. When the sequence buffer matches a combo, the jutsu
is triggered.

**Combos defined:** Katon Fireball (3 signs), Shadow Clone (3 signs), Chidori (3 signs)

**Key concepts:** Sequence matching, tuple-keyed dictionaries, combo systems.

---

### Stage 7 - Visual Effects
Added a full-screen colour flash overlay when a jutsu combo triggers, with a smooth
alpha fade over ~2 seconds. Each jutsu has a unique colour theme. Jutsu name is
displayed centred on screen during the effect. Sequence auto-resets after trigger.

**Key concepts:** Alpha blending, overlay compositing, frame-based timers,
OpenCV drawing utilities.

### Stage 8 - Reliability Fixes
Rotation-invariant finger detection (distances from the wrist instead of screen
height), thumb ignored by every sign, tolerance for brief tracking dropouts, a 2s
timeout between signs, and webcam-free unit tests.

---

### Stage 9 - Two Hands, Effects and Game Mode
Up to two hands, each read as its own sign and combined left-to-right
(`Ushi+Ushi`). Combos of any length carry their own name, colour and sound.
Particle bursts and generated sounds on trigger, and a 60-second challenge mode.

---

## Project Architecture

```
jutsu_code/
├── main.py            — Entry point, camera loop, UI rendering
├── hand_detector.py   — MediaPipe wrapper, landmark extraction, finger states
├── signs.py           — Sign pattern definitions and matching logic
├── stabilizer.py      — Frame buffer smoothing for sign confirmation
├── sequence.py        — Sign change detection and sequence buffer
├── combos.py          — Combo definitions and matching
├── effects.py         — Particle bursts and generated sounds
├── game.py            — Timed challenge mode
└── test_logic.py      — Webcam-free tests
```

---

## Key Engineering Decisions

- **No ML training** — all gesture recognition is rule-based, making it fast,
  lightweight, and fully explainable.
- **Modular architecture** — each concern (detection, stabilization, sequencing,
  actions) lives in its own file, making the system easy to extend.
- **Wildcard pattern matching** — allows unreliable fingers to be excluded per sign
  without breaking the matching system.
- **Frame-buffer stabilization** — solves real-world flickering without adding latency
  that would feel unresponsive to the user.

---

## Skills Demonstrated

- Computer vision pipeline design
- Real-time video processing
- Gesture recognition system architecture
- Python OOP and modular design
- Virtual environment and dependency management
- Debugging hardware/library compatibility issues
- Iterative, phase-based software development

---

## Current Status

Stages 1–9 complete: four combos (one two-handed) with flash, particle and sound
effects, plus a challenge mode. Next: an extended sign library, sprite overlays,
and multiplayer.

---

*Built in 2026*
