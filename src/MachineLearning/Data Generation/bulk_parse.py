# bulk_parse.py — zero-arg runner
import os, sys, re, subprocess
import pandas as pd

#CONFIG
BASE_DIR = r"C:\Users\nickl\OneDrive\Documents\Computer-Vision-Powered-AI-Poker-Coach\ML Model Work\poker-hand-histories\data\handhq"
OUT_DIR  = os.path.join(os.getcwd(), "out")  # or any folder you like

HAND_PARSER = r"C:\Users\nickl\OneDrive\Documents\Computer-Vision-Powered-AI-Poker-Coach\ML Model Work\hand_parser.py"
STAKE_RE = re.compile(r"^(?:\d+(?:\.\d+)?)$")  # '0.5', '1', '2', '4', etc.

def find_stake_dirs(base_dir: str):
    stake_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if STAKE_RE.match(d):
                stake_dirs.append(os.path.join(root, d))
    return sorted(stake_dirs)

def rel_out_path(stake_dir: str):
    rel = os.path.relpath(stake_dir, BASE_DIR)
    safe = rel.replace("\\", "__").replace("/", "__")  # flatten to avoid many subfolders
    return os.path.join(OUT_DIR, f"parsed__{safe}.csv")

def run_parser(stake_dir: str, out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    print(f"→ Parsing {stake_dir}")
    # First try phhs
    cmd = [sys.executable, HAND_PARSER, "-i", stake_dir, "-o", out_csv, "--glob", "*.phhs"]
    subprocess.run(cmd, check=True)

def main():
    if not os.path.isfile(HAND_PARSER):
        print(f"[!] hand_parser.py not found at: {HAND_PARSER}")
        sys.exit(1)
    if not os.path.isdir(BASE_DIR):
        print(f"[!] BASE_DIR does not exist: {BASE_DIR}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    stake_dirs = find_stake_dirs(BASE_DIR)
    if not stake_dirs:
        print("No stake directories found under BASE_DIR.")
        sys.exit(1)

    # Parse each stake dir
    shards = []
    for sd in stake_dirs:
        out_csv = rel_out_path(sd) 
        run_parser(sd, out_csv)
        shards.append(out_csv)

if __name__ == "__main__":
    main()

