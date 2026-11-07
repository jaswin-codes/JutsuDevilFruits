# JutsuDevilFruits — User Guide

> How to use the system, what signs to perform, and how to trigger each jutsu.

---

## Setup

1. Open VS Code in the project folder
2. Activate the virtual environment in the terminal:
   ```
   .\venv\Scripts\Activate
   ```
3. Run the program:
   ```
   python main.py
   ```
4. A window opens with a start menu over your webcam feed: press **1** Naruto or
   **2** One Piece. Naruto then asks for the mode: **1** free play, **2** challenge,
   **3** monster battle or **4** duel. **0** shows the hand-sign chart: a photo of every sign
   with its name. Any key goes back
5. Press **M** to go back to the menu and **Q** to quit at any time. In a Naruto mode, **H**
   opens the sign chart and **H** closes it. The game is paused while it is open

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
MIT licence). It knows the 12 zodiac signs from the anime, plus prayer hands. To see a photo of
each sign, press **0** in the Naruto menu or **H** during a game. The photos come from that
repository's README (MIT, see `assets/signs/SOURCE.md`).

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

### 🪵 Substitution Jutsu
**Sequence:** `Tatsu → Ne` (Dragon, Rat). Our own pick: Tatsu is the one sign no other jutsu
uses, so it can't fire by accident halfway through another jutsu.

**Effect:** For 4 seconds after you cast it, the next hit lands on a wooden log instead of you,
in a puff of smoke. In battle mode the first monster to reach you mauls the log; in a duel it
takes any jutsu (one hit, then it's used up). In free play the log appears on the spot.
Cooldown 8 s.

---

### 🪨 Summoning Jutsu
**Sequence:** `I → Inu → Tori → Saru → Hitsuji` (Boar, Dog, Bird, Monkey, Ram), then **slap a palm
down** toward the bottom of the screen within 4 seconds ("SLAP A PALM DOWN!" shows while it waits).

**Effect:** A smoke burst where your hand lands, and a boulder beast with crystal shoulders and
three amber eyes rises out of it for 4 seconds. The slap has to be quick: a hand lowered slowly
doesn't count. In battle the creature crushes the monsters near it; in a duel it charges your
opponent.

---

### 👁️ Sharingan
**Sequence:** none. **Cover one eye with your hand, then lower it** (hold it there for about
half a second, no longer than 3 seconds).

**Effect:** Your eyes turn red with three spinning tomoe for 5 seconds. Face the camera with
your eyes open so the face tracker can find your irises. In battle the monsters slow to half
speed; in a duel every one of your opponent's jutsu takes 3 seconds longer to be ready.

---

## Game Mode

Run `python main.py --game`. A target jutsu and its signs appear at the bottom, with a small
photo of each sign in order above the names.
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
| Substitution | Dodge | The next monster to reach you within 4 s hits a log instead. | 8 s |
| Summoning | Area | Slap a palm down toward a side: the creature lands there and deals 150 to every monster within a quarter screen of it. | 20 s |
| Sharingan | Slow | Monsters walk at half speed for 5 s. | 15 s |

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
each player's signs, hands and jutsu are read from their own half only.

A match is **best of 3 rounds**. Each round starts with a 3-2-1-FIGHT countdown (jutsu do
nothing before FIGHT), both players at 100 HP and every jutsu ready. A round ends when one
player reaches 0 HP; the next starts by itself after 3 seconds. The round and score are shown
between the HP bars. After the match, press **R** for a rematch.

| Jutsu | How it hits | Damage | Cooldown |
|---|---|---|---|
| Fireball | Flies across to the other player and hits once when it reaches them. Facing the camera aims it straight at your opponent; turning your head away sends it off your own side. | 40 (the slowest jutsu to perform hits hardest) | 1 s |
| Chidori | Reach toward the centre line: hits while the lightning ball touches it. | 20/s, max 35 per cast | 5 s |
| Rasengan | Same as Chidori; a bigger ball (hands further apart) hits faster. | up to 50/s, max 35 per cast | 10 s |
| Shadow Clone | Your clones take every hit for 3 s, including a fireball, which is used up. | - | 12 s |
| Hiding with Camouflage | The opponent's fireball can't find you for 5 s (Chidori and Rasengan still can). | - | 15 s |
| Substitution | The next hit within 4 s, of any jutsu, lands on a log instead. | - | 8 s |
| Summoning | The creature charges your opponent once, as soon as it appears. Shadow Clone and Substitution block it. | 30 | 20 s |
| Sharingan | Every opponent jutsu, ready or cooling down, is ready 3 s later. | - | 15 s |

Each player's HP bar and cooldowns are in their bottom corner; a red edge round your half
means you were hit.

Tips:
- Stand in the middle of your own half and keep your hands on your side of the line;
  hands that cross the line are counted as the other player's.
- Leave some space between you: the sign model was trained on one person per picture, so
  two people overlapping can confuse it.
- No single jutsu wins a round: mix fast Chidori pokes with a well-timed Fireball, and keep
  Shadow Clone or Substitution ready for the opponent's Fireball.

---

## One Piece Mode

Pick **2 One Piece** in the start menu, then a devil fruit: **1 Gomu Gomu**, **2 Mera Mera** or
**3 Goro Goro** (M goes back to the worlds).

### Gomu Gomu (rubber)

Pick **1 Gomu Gomu**, or run `python main.py --one-piece`. No hand signs here:
your body is rubber, like Luffy's. Only your hands are tracked (the gears also find your body
outline, to tint and puff it up).

| Move | How | What happens |
|------|-----|--------------|
| Gomu Gomu no Pistol | Make a fist, side-on to the camera | The fist shoots out the way your hand points, the arm stretching behind it, until it reaches the edge of the screen. Open your hand and it snaps back. |
| Rubber face | Pinch your mouth corner (or cheek, ear...) between thumb and index finger and pull | The skin stretches out twice as far as your fingers move, as a round tube, wide on your face and narrow at your fingers. Let go and it snaps back with a wobble. |
| Gomu Gomu no Gatling | Make a fist and pump it back and forth fast (side to side or up and down) | A fan of fists flies out around the Pistol, each one shooting in and out. Stops half a second after you stop pumping. |
| Gear Second | Hold a V sign (index and middle finger up, ring and pinky curled) for half a second | For 10 seconds your body glows pink-red with steam rising off it, and Pistol flies out twice as fast. |
| Gear Fourth | Hold up four fingers with the thumb tucked across your palm for half a second | For 10 seconds your hands, forearms and chest puff up like a balloon and Pistol fists are 1.6x bigger. Ends Gear Second (and the other way round). |

Both hands work at once: pull both cheeks, or fire a Pistol while pulling your face.

**Tips:** the other fingers can curl while you pinch; just keep the index finger reaching out to
the thumb (an index curled into the palm reads as a fist). Hold the fist side-on: knuckles facing the camera points it up instead of sideways. For the Gatling, move
the whole fist across the screen; punching straight at the camera is not seen as a pump. The
gear and its seconds left show next to ONE PIECE at the bottom.

### Mera Mera (fire)

Pick **2 Mera Mera**, or run `python main.py --mera`.

| Move | How | What happens |
|------|-----|--------------|
| Burning fist | Make a fist | Flames rise off the fist and follow it wherever it goes. |
| Hiken | Punch the fist straight at the camera, fast | A fire blast swells at your fist and rushes at the screen. Punch again once it has burnt out. |

**Tips:** the punch is read from your knuckles growing on screen (about a third bigger within a
quarter of a second), so start with the fist an arm's length away and punch quickly. Slowly
bringing the fist closer does nothing.

### Goro Goro (lightning)

Pick **3 Goro Goro**, or run `python main.py --goro`.

| Move | How | What happens |
|------|-----|--------------|
| Vari | Point your index finger up, the other fingers curled | Lightning crackles around your hand for as long as you point. |
| El Thor | Raise both open hands, fingers up, and hold them for a third of a second | A forked bolt strikes from the top of the screen down between your hands. Lower your hands and raise them again for another. |

**Tips:** keep both hands fully in view with all fingers spread for El Thor; a closed finger
breaks the hold.

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

*JutsuDevilFruits user guide (2026)*
