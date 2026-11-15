# Contributing

Bug reports, gesture tuning notes and new-power ideas are all welcome.

1. Use Python 3.11 and `pip install -r requirements.txt`.
2. Run `python -m pytest -q` before opening a pull request. The tests need no webcam.
3. New sign combos are one line in `combos.py`. Keep game logic free of camera and drawing code so it stays testable.
4. Keep pull requests small and say which mode you tried it in.
