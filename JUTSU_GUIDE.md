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
2. Keep it **steady** until the sign name appears on screen (about half a second)
3. **Change** to the next sign in the sequence — holding the same sign does nothing
4. Complete the full combo sequence to trigger the jutsu
5. The jutsu effect plays and its name appears at the top
6. The sequence resets automatically — you can go again immediately

---

## Tips for Best Detection

- Make each sign with **both hands together**, close to your chest and facing the camera
- Use **good lighting**; avoid backlighting (don't sit with a window behind you)
- Sit **roughly 50–100 cm** from the camera so your hands and face are both in view
- Move **deliberately** between signs; don't rush
- The green box shows where the model sees your sign, and "Raw" shows its guess

---

## The Signs

Signs are read by a trained model (YOLOX-Nano from
[Kazuhito00/NARUTO-HandSignDetection](https://github.com/Kazuhito00/NARUTO-HandSignDetection),
MIT licence). It knows the 12 zodiac signs from the anime, plus prayer hands. For photos of each
sign, see the sign chart in that repository's README.

| Name on screen | Sign |
|---|---|
| Ne | Rat |
| Ushi | Ox |
| Tora | Tiger |
| U | Hare |
| Tatsu | Dragon |
| Mi | Snake |
| Uma | Horse |
| Hitsuji | Ram |
| Saru | Monkey |
| Tori | Bird |
| Inu | Dog |
| I | Boar |
| Gassho | Prayer hands (palms pressed together) |

---

## Jutsu Combos

> Perform each sign in order, holding each one until it confirms before moving to the next.
> The orders follow the anime.

---

### 🔥 Katon — Fireball Jutsu
**Sequence:** `Mi → Tora → Saru → I → Uma → Tora` (Snake, Tiger, Monkey, Boar, Horse, Tiger)

**Effect:** A fireball swells at your mouth, then launches. Face the camera to blast it at the screen, or turn your head
left or right to send it flying that way. Open your mouth wide while it gathers for a bigger fireball.
Aim and size lock in when it launches (half a second after the jutsu fires).

---

### 👥 Shadow Clone Jutsu
**Sequence:** `Hitsuji → Mi → Tora` (Ram, Snake, Tiger)

**Effect:** Four copies of you pop up beside you in a puff of smoke for 3 seconds. Works best with a plain background.

---

### ⚡ Chidori
**Sequence:** `Ushi → U → Saru` (Ox, Hare, Monkey)

**Effect:** A crackling white-blue lightning ball sits in your palm and follows it until you lower your hand.
It charges up, growing for 3 seconds while you hold it. Bring your hand closer to the camera to make it bigger too.

---

### 🌀 Rasengan
**Sequence:** `Gassho` (prayer hands). Rasengan has no hand signs in the anime.

**Effect:** A spinning blue chakra sphere forms between your hands and follows them until you lower them.
Pull your hands apart to grow it, or bring them together to shrink it.

---

### 👻 Hiding with Camouflage (invisibility)
**Sequence:** `Ne → Inu → Tori` (Rat, Dog, Bird). These signs are our own pick, not from the anime.

**Effect:** You fade out for 8 seconds: the app paints the room behind you over your body, with a faint
heat-haze ripple where you stand. In battle mode the monsters lose track of you and stand still for 5 seconds.

**Setup:** the app has to have seen the room without you. It learns the background all the time, so
step out of the picture for a second (or move to the other side) before you use it. Spots it has never
seen empty show as a blur. Keep the camera still: if it moves, step out again so it relearns.

---

## Game Mode

Run `python main.py --game`. A target jutsu and its signs appear at the bottom.
Perform it to score a point and get a new target. You have 60 seconds;
press **R** to play again.

---

## Battle Mode

Run `python main.py --battle`. Your webcam image is the arena and you are the hero:
monsters walk in from both sides of the screen toward the middle. Each one that reaches
you costs 15 HP (you have 100). Clear a wave to start the next, which brings more,
tougher and faster monsters. Press **R** to fight again after a defeat.

| Jutsu | Role | How it hits | Cooldown |
|---|---|---|---|
| Fireball | Ranged | Turn your head toward a side: hits the nearest monster on that side. Facing the camera hits the nearest monster on either side. | 1 s |
| Chidori | Melee | Damages any monster your palm's lightning ball touches. Reach toward them. | 5 s |
| Rasengan | Charged melee | Same as Chidori, but a bigger ball (hands further apart) hits harder. | 10 s |
| Shadow Clone | Block | For 3 s your clones hold the monsters back and you take no damage. | 12 s |
| Hiding with Camouflage | Escape | Monsters lose track of you and stand still for 5 s. | 15 s |

A jutsu still on cooldown shows a "cooling down" notice and does nothing; its timer is at
the bottom right. Your HP, wave and score are at the bottom left.

Tips:
- Stand in the middle of the picture: the "you" zone is the centre of the screen.
- Wave 1 monsters take about 9 s to reach you, enough for the six-sign Fireball.
  Start it early, and save Shadow Clone for when several monsters are close.
- Keep your hands low to reach the monsters with Chidori and Rasengan.

---

## Duel Mode

Run `python main.py --duel`. Two players stand side by side in front of one webcam; the
screen splits down the middle. **Player 1 is the left half, player 2 the right half**, and
each player's signs, hands and jutsu are read from their own half only. Both start with
100 HP; the round ends when one reaches 0. Press **R** for a new round.

| Jutsu | How it hits | Damage | Cooldown |
|---|---|---|---|
| Fireball | Flies across to the other player and hits once when it reaches them. Facing the camera aims it straight at your opponent; turning your head away sends it off your own side. | 30 | 1 s |
| Chidori | Reach toward the centre line: hits while the lightning ball touches it. | 20/s, max 35 per cast | 5 s |
| Rasengan | Same as Chidori; a bigger ball (hands further apart) hits faster. | up to 50/s, max 35 per cast | 10 s |
| Shadow Clone | Your clones take every hit for 3 s, including a fireball, which is used up. | - | 12 s |
| Hiding with Camouflage | The opponent's fireball can't find you for 5 s (Chidori and Rasengan still can). | - | 15 s |

Each player's HP bar and cooldowns are in their bottom corner; a red edge round your half
means you were hit.

Tips:
- Stand in the middle of your own half and keep your hands on your side of the line;
  hands that cross the line are counted as the other player's.
- Leave some space between you: the sign model was trained on one person per picture, so
  two people overlapping can confuse it.
- No single jutsu wins a round: mix fast Chidori pokes with a well-timed Fireball, and keep
  Shadow Clone ready for the opponent's Fireball.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Sign not being detected | Hold both hands steadier, closer to the centre of the frame |
| Sign flickering between two values | Improve lighting, move hand closer |
| Combo not triggering | Make sure each sign **confirms** before moving to next |
| Camera not opening | Run `python main.py --camera 1` to try another webcam |
| Program crashes on start | Make sure venv is activated before running |

---

*Jutsu Code user guide (2026)*
