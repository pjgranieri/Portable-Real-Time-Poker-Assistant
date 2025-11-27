# Computer-Vision-Powered-AI-Poker-Coach

## Heads-Up Evaluation Harness

We provide a script to benchmark the LiveHandTracker policy against simple rule-based opponents in heads-up poker using PyPokerEngine. It reports win rate in big blinds per 100 hands (BB/100) with 95% confidence intervals and saves a summary CSV.

Location:
- `scripts/evaluate_headsup_matchups.py`

Key details:
- Opponents: NIT, STATION, WHALE styles (from `ML Model Work/Data Generation/pokerBotDataCreator.py`).
- Seats alternate every session to remove position bias.
- Sizing map: policy `raise_s`/`raise_m`/`raise_l` → ~0.33/0.66/1.10 pot within engine constraints.
- Per-hand results are recorded at the end of each hand using stack deltas.

Quick start (PowerShell):

```powershell
python scripts/evaluate_headsup_matchups.py --hands 2000 --styles NIT --seed 1
```

Full evaluation (PowerShell):

```powershell
python scripts/evaluate_headsup_matchups.py --hands 50000 --styles NIT STATION WHALE --seed 42
```

Output:
- Console summary per opponent
- CSV at `runs/eval_matchups/matchup_summary.csv`

Notes:
- The script adjusts `sys.path` at runtime to import `LiveHandTracker` and `SuperBot` from the project layout.
- Ensure `pypokerengine` and `treys` are installed in your Python environment.