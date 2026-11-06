# JutsuDevilFruits: the whole project in one file

Written 2026-11-06 to replace 9 separate chats. Read this first in any new session, then
README.md (setup and features) and JUTSU_GUIDE.md (how to perform every move).

## 1. What it is

A Python 3.11 webcam app (OpenCV, MediaPipe 0.10.21, onnxruntime). You do anime powers in
front of your webcam and the effects are drawn over the live image.
- **Naruto world:** real hand signs read by a trained model fire jutsu. Five modes: free
  play, challenge, monster battle, duel, and a sign chart.
- **One Piece world:** devil-fruit powers from plain hand gestures, no signs.

- Repo: https://github.com/jaswin-codes/JutsuDevilFruits (folder `C:\Users\jaswi\Downloads\jutsu_code_stage1`)
- venv: `venv\Scripts\python.exe` in that folder. Worktrees have no venv of their own; from
  one, use `..\..\..\venv\Scripts\python.exe`.
- Run: `venv\Scripts\python.exe main.py` (start menu). Tests: `venv\Scripts\python.exe -m pytest -q`.
  CI runs the same tests on every push and PR (`.github/workflows/tests.yml`).

## 2. Status right now (read this before doing anything)

| Where | What | State |
|---|---|---|
| `origin/main` (7889d94) | Everything up to the One Piece gears and the sign chart (PR #9) | merged |
| branch `claude/inspiring-goldberg-3322f6` | Devil-fruit menu, Mera Mera, Goro Goro, docs, this file | **local only, not pushed, no PR**. Cut from current main, 85 tests pass |
| branch `claude/silly-stonebraker-fa85db` | Summoning Jutsu and Sharingan (5 commits) | **local only, not pushed**. Cut from the OLD main (ee632e0, before One Piece), so merging it will conflict in `main.py`, `README.md` and `JUTSU_GUIDE.md` |
| main checkout | on `feat/one-piece` (already merged) | switch to `main` and pull |
| `C:\Users\jaswi\Downloads\jutsu_invisibility_test` | old worktree of `feat/invisibility` (merged) | safe to remove |

Nothing in Mera Mera, Goro Goro, Summoning or Sharingan has been tried on a real webcam yet.
The One Piece gears haven't been tried on one either.

## 3. Timeline

- **2026-07-06, v1 "Jutsu Code" (stage 1):** MediaPipe hand landmarks, hand-written finger
  rules for each sign, a stabilizer and a sequence detector, combos, simple effects. MIT licence.
- **2026-10-25:** LICENSE re-encoded as UTF-8; requirements.txt with mediapipe pinned.
- **2026-11-04:**
  - Reliability fixes (PR #1): tracking dropouts, thumb ignored, rotation-invariant fingers,
    a 2 s gap discards stale signs, and webcam-free tests.
  - Two-hand signs, particles and sounds, and the 60 s challenge mode (PRs #2 and #3).
  - Anime effects: Chidori, Rasengan, shadow clones (selfie segmentation), Fireball (face mesh).
  - **Big switch:** the finger rules were replaced by a trained YOLOX-Nano sign model
    (Kazuhito00/NARUTO-HandSignDetection, MIT), with canon sign orders.
- **2026-11-05:**
  - Rasengan sized by hand spread, Chidori charges up, Fireball aimed by head turn.
  - Monster battle mode, invisibility (learned background plate), two-player split-screen
    duel (best of 3), substitution jutsu, start menu, CI (PRs #4 to #8).
- **2026-11-06:**
  - One Piece world with Luffy: Pistol, rubber face pull, Gatling, Gear Second, Gear Fourth.
  - Sign chart with photos (0 in the menu, H mid-game, pauses the clocks).
  - App renamed **JutsuDevilFruits** (PR #9).
  - Then, on the unmerged branches: Summoning and Sharingan; the devil-fruit picker with Mera
    Mera and Goro Goro.

## 4. Modes and controls

Start menu: **1 Naruto, 2 One Piece**. M goes back a menu and Q quits everywhere.
`--camera N` picks another webcam.

**Naruto** menu: 1 Free play (`--free`), 2 Challenge (`--game`), 3 Monster battle (`--battle`),
4 Duel (`--duel`), 0 Hand-sign chart. H opens the chart mid-game and pauses everything.
- **Challenge:** 60 s. Perform the target jutsu (its sign photos are shown) for points. R restarts.
- **Battle:** monsters walk in from both sides and cost 15 HP each (you have 100). Waves get
  harder, every jutsu has a role and a cooldown, and R restarts.
- **Duel:** two players, left and right halves of the screen. HP, cooldowns, blocks, best of
  3 with a countdown, R rematch.

**One Piece** menu (once the devil-fruit branch merges): 1 Gomu Gomu (`--one-piece`),
2 Mera Mera (`--mera`), 3 Goro Goro (`--goro`). Each fruit is its own mode.
- **Gomu Gomu:**
  - Fist = Pistol; pump the fist = Gatling.
  - Pinch your face and pull = rubber face.
  - Hold a V sign = Gear Second (pink, steam, faster).
  - Four fingers with the thumb tucked = Gear Fourth (puffed up).
- **Mera Mera:** fist = burning fist. Fast punch at the camera (knuckle width grows 1.35× in
  0.25 s) = HIKEN fire blast.
- **Goro Goro:** one finger up = VARI lightning. Both open hands held up for 0.3 s = EL THOR,
  a bolt from the top of the frame.

## 5. Hand signs and jutsu

Signs (from the model): Ne (rat), Ushi (ox), Tora (tiger), U (hare), Tatsu (dragon), Mi (snake),
Uma (horse), Hitsuji (ram), Saru (monkey), Tori (bird), Inu (dog), I (boar), Gassho (prayer hands).
Photos are in `assets/signs` (MIT, see `SOURCE.md`).

| Jutsu | Signs | Effect | Battle role |
|---|---|---|---|
| Katon Fireball | Mi > Tora > Saru > I > Uma > Tora | Fireball from the mouth, aimed by head turn | ranged, 1 s |
| Shadow Clone | Hitsuji > Mi > Tora | Clones of you | block, 12 s |
| Chidori | Ushi > U > Saru | Lightning ball in the palm, charges up | melee, 5 s |
| Rasengan | Gassho | Sphere between the palms; spread your hands to grow it | charged melee, 10 s |
| Hiding with Camouflage | Ne > Inu > Tori (our pick) | Invisibility | escape, 15 s |
| Substitution | Tatsu > Ne (our pick) | The next hit lands on a log | dodge, 8 s |
| Summoning *(unmerged)* | I > Inu > Tori > Saru > Hitsuji, then slap a palm down within 4 s | A boulder beast rises | area, 20 s |
| Sharingan *(unmerged)* | Cover one eye, then lower the hand | Red eyes with spinning tomoe | slows monsters / opponent, 15 s |

Add a combo with one line in `combos.py`.

## 6. Code map

- **Naruto pipeline:**
  - `main.py`: menus, the Naruto loop, flags.
  - `sign_model.py`: ONNX sign model, one run split per screen half.
  - `stabilizer.py`, `sequence.py`, `combos.py`: sign → combo.
- **Effects and game logic:**
  - `anime_effects.py`: pure draw functions (chidori, rasengan, clones, fireball, invisibility,
    log, bolt).
  - `effects.py`: particles and sounds.
  - Pure logic: `game.py`, `battle.py` and `duel.py`. Their drawing is in `battle_view.py` and
    `duel_view.py`.
  - `sign_chart.py`: sign photos.
- **One Piece:**
  - `one_piece.py`: Gomu Gomu, plus the shared `gesture()` / `fingers_out()`.
  - `mera.py`, `goro.py`: one module per fruit.
- **Tests:** `test_*.py`, no webcam needed. `test_menu.py` drives `main()` with a fake camera,
  keys, clock and hand detector.

## 7. Decisions you made (and why)

- **Reuse a trained sign model; don't train one.** Real two-hand signs interlock the fingers,
  and finger rules can't read that.
- **Canon sign orders** from the anime. Where canon has none, we picked unused signs so no jutsu
  fires halfway through another one.
- **Keep logic and drawing separate.** Pure logic is unit-tested on landmark lists, and the
  draw functions are rendered to PNG for checking.
- **Stack:** Python 3.11 and mediapipe 0.10.21 (no 3.13 support). Never add `opencv-python`:
  mediapipe brings `opencv-contrib-python`.
- **Expand beyond Naruto.** That led to the world menu and then the rename to JutsuDevilFruits.
- **Each devil fruit is its own mode,** not more gestures on Luffy. Fist, V and four fingers are
  already taken there.
- **Effects are drawn in code only:** text names, no copyrighted images. Sign photos are the
  exception (MIT, credited).
- **One window name** everywhere: "JutsuDevilFruits".
- **Two players in a duel share one webcam** (split screen) rather than going online.

## 8. Routes taken and dropped

- **Finger-rule signs (v1):** dropped. They were unreliable and couldn't read two-hand signs.
- **Rotation-invariant fingers, thumb ignored:** kept until the model replaced them.
- **Full-resolution fireball shading:** too slow (~150 ms). It is now shaded at 1/4 size and
  scaled up, keeping effects under ~20 ms per frame.
- **Pinch detection:** first required the thumb to be visible. Now it is the index reaching the
  thumb, with longer grace frames, because the other fingers hide the thumb near the face.
- **Punch detection:** the brief assumed Pistol used palm growth; it doesn't. Mera uses knuckle
  width, because wrist-to-knuckle length shrinks when the fist points at the camera.

## 9. Working rules (from you)

- **Git:**
  - Conventional Commits, one concern per commit. Only commit when the tests pass.
  - Commit locally; push only when asked.
  - Stage files by name. Never commit `HANDOFF.md`, `.env`, build output or the
    "Open Notebook.onetoc2" OneNote files (they sit in several folders).
  - `git config user.name` is unset in this repo, so commits use
    `git -c user.name="Jaswin Chinthala" -c user.email=jaswinchinthala08@gmail.com commit ...`.
    Setting it globally would fix this for good.
- **How to work:** don't be a yes-man; say when a plan has a weakness.
- **Calendar:** put deadlines in Google Calendar.
- **Models:** Sonnet for routine work, Opus for hard work.
- **Handoffs:** hand off at about 200K tokens of context, with a HANDOFF.md.
- **Plugins:** turn ECC on per project for big projects:
  `claude plugin enable ecc@ecc --scope project`.

## 10. Next steps, in order

1. **Main checkout:** `git checkout main` and `git pull`.
2. **Devil fruits:** push `claude/inspiring-goldberg-3322f6` (or rename it to
   `feat/devil-fruits`), open a PR and merge.
3. **Summoning and Sharingan:** rebase `claude/silly-stonebraker-fa85db` onto the new main and
   resolve the conflicts. `main.py` must keep both the world/fruit menus and the summon slap
   and Sharingan code. In the docs, keep both the One Piece sections and the new jutsu sections.
   Run the tests, then push and open a PR.
4. **Webcam test session:** try every move on a real camera and tune the constants at the top
   of each module:
   - Mera: `PUNCH_GROWTH`, `PUNCH_WINDOW`
   - Goro: `RAISE_HOLD`, `POINT_UP`
   - Gomu: `PUMP_DIST`, `THUMB_IN`, `GEAR4_PUFF`
   - Summoning: the slap speed
5. **Cleanup:** remove the `jutsu_invisibility_test` worktree and the merged local branches.

## 11. Ideas for later

- **Fruits and battles:** One Piece fruits in battle and duel mode (Hiken as a ranged attack,
  El Thor as an area attack).
- **More fruits:** Hie Hie (ice), Bara Bara (split hands), Gura Gura (screen shake).
- **Naruto:** more jutsu (Rasenshuriken, Amaterasu with the Sharingan, Water Dragon).
- **Records:** high scores saved to a file, and a recording/screenshot key for sharing clips.
- **Mixed duel:** a Naruto player against a One Piece player.
- **Packaging:** an exe with PyInstaller, so friends can play without Python.
