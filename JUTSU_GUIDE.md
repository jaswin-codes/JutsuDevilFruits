# Jutsu Code — User Guide

> How to use the system, what signs to perform, and how to trigger each jutsu.

---

## Setup

1. Open VS Code in the `jutsu_code` folder
2. Activate the virtual environment in the terminal:
   ```
   .\venv\Scripts\Activate
   ```
3. Run the program:
   ```
   python main.py
   ```
4. A window will open showing your webcam feed
5. Press **Q** to quit at any time

---

## How It Works

1. Hold a sign in front of the camera
2. Keep it **steady** until the sign name appears on screen (~1/3 of a second)
3. **Change** to the next sign in the sequence — holding the same sign does nothing
4. Complete the full combo sequence to trigger the jutsu
5. The jutsu effect plays and its name appears at the top
6. The sequence resets automatically — you can go again immediately

---

## Tips for Best Detection

- Keep your hand **facing the camera** (palm toward you or slightly angled)
- Use **good lighting** — avoid backlighting (don't sit with a window behind you)
- Keep your hand **roughly 30–60cm** from the camera
- Move **deliberately** between signs — don't rush
- If a sign isn't being detected, **re-centre** your hand and hold it steadier

---

## The Five Signs

> Each sign is shown as a finger pattern: `[Thumb | Index | Middle | Ring | Pinky]`
> **UP = 1 (finger extended)** | **DOWN = 0 (finger curled)**

---

### 🐯 Tora (Tiger)
**Pattern:** `[any | 1 | 1 | 0 | 0]`

```
Thumb:  ANY
Index:  UP    ☝️
Middle: UP    ✌️
Ring:   DOWN  ✊
Pinky:  DOWN  ✊
```
**How to make it:** Peace sign / scissors hand. Index and middle fingers extended,
all others curled in. Like the classic ✌️ sign.

---

### 🐂 Ushi (Ox)
**Pattern:** `[any | 1 | 1 | 1 | 1]`

```
Thumb:  ANY
Index:  UP    ☝️
Middle: UP    🖕
Ring:   UP    💍
Pinky:  UP    🤙
```
**How to make it:** Open palm facing the camera. All five fingers fully extended
and spread apart. Like a stop sign or a high five.

---

### 🐭 Ne (Rat)
**Pattern:** `[any | 1 | 0 | 0 | 0]`

```
Thumb:  ANY
Index:  UP    ☝️
Middle: DOWN  ✊
Ring:   DOWN  ✊
Pinky:  DOWN  ✊
```
**How to make it:** Point upward with only your index finger. All other fingers
curled in tight. Classic pointing gesture ☝️.

---

### 🐕 Inu (Dog)
**Pattern:** `[any | 0 | 0 | 0 | 0]`

```
Thumb:  ANY
Index:  DOWN  ✊
Middle: DOWN  ✊
Ring:   DOWN  ✊
Pinky:  DOWN  ✊
```
**How to make it:** Closed fist. All fingers curled in, thumb tucked.
Show the back of your fist to the camera ✊.

> **Note:** Thumb is a wildcard for every sign — it doesn't matter if your thumb is
> slightly out, the fist will still be detected.

---

### 🐦 Tori (Bird)
**Pattern:** `[any | 1 | 0 | 0 | 1]`

```
Thumb:  ANY
Index:  UP    ☝️
Middle: DOWN  ✊
Ring:   DOWN  ✊
Pinky:  UP    🤙
```
**How to make it:** Rock on / metal horns 🤘. Index finger and pinky extended,
middle and ring fingers curled down, thumb tucked in.

---

## Jutsu Combos

> Perform each sign in order, holding each one until it confirms before moving to the next.

---

### 🔥 Katon — Fireball Jutsu
**Sequence:** `Ne → Tora → Inu`

```
Step 1: ☝️  Ne   (index finger up)
Step 2: ✌️  Tora (peace sign)
Step 3: ✊  Inu  (fist)
```
**Effect:** A fireball swells at your mouth, then rushes at the camera. Face the camera so your mouth is found.

---

### 👥 Shadow Clone Jutsu
**Sequence:** `Ushi → Ne → Tora`

```
Step 1: 🖐️  Ushi (open palm)
Step 2: ☝️  Ne   (index finger up)
Step 3: ✌️  Tora (peace sign)
```
**Effect:** Four copies of you pop up beside you in a puff of smoke for 3 seconds. Works best with a plain background.

---

### ⚡ Chidori
**Sequence:** `Tori → Inu → Ushi`

```
Step 1: 🤘  Tori (rock on)
Step 2: ✊  Inu  (fist)
Step 3: 🖐️  Ushi (open palm)
```
**Effect:** A crackling white-blue lightning ball sits in your palm and follows it until you lower your hand.

---

### 🌀 Rasengan (two hands)
**Sequence:** `Ushi+Ushi → Tora+Tora`

```
Step 1: 🖐️🖐️  Ushi with both hands
Step 2: ✌️✌️  Tora with both hands
```
**Effect:** A spinning blue chakra sphere forms between your hands and follows them until you lower them.

> With two hands in view, the sign is read as `Left+Right` (your own hands).
> Both hands must show a known sign, or nothing registers.

---

## Game Mode

Run `python main.py --game`. A target jutsu and its signs appear at the bottom.
Perform it to score a point and get a new target. You have 60 seconds;
press **R** to play again.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Sign not being detected | Hold hand steadier, face palm more toward camera |
| Sign flickering between two values | Improve lighting, move hand closer |
| Combo not triggering | Make sure each sign **confirms** before moving to next |
| Thumb always showing UP | Normal — thumb is ignored by every sign |
| Camera not opening | Run `python main.py --camera 1` to try another webcam |
| Program crashes on start | Make sure venv is activated before running |

---

*Jutsu Code user guide (2026)*
