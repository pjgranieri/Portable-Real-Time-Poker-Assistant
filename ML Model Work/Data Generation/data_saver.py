import os, glob, math, argparse
import pandas as pd
import numpy as np

# CONFIG 
OUT_DIR_DEFAULT    = os.path.join(os.getcwd(), "out")    
SHARDS_DIR_DEFAULT = os.path.join(os.getcwd(), "Poker Data (Real)") 
N_SHARDS_DEFAULT   = 500
HANDS_PER_SHARD    = 1000
RANDOM_SEED        = 42


def load_all_real_rows(out_dir: str) -> pd.DataFrame:
    csvs = sorted(glob.glob(os.path.join(out_dir, "parsed__*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No parsed__*.csv found in {out_dir}.")
    print(f"Concatenating {len(csvs)} CSVs from {out_dir}")
    dfs = []
    for i, p in enumerate(csvs, 1):
        try:
            dfi = pd.read_csv(p, low_memory=False)
            dfs.append(dfi)
        except Exception as e:
            print(f"Skipping {p}: {e}")
        if i % 25 == 0:
            print(f" {i}/{len(csvs)} files")
    df = pd.concat(dfs, ignore_index=True)
    return df

def shard_by_hands(df: pd.DataFrame,
                   shards_dir: str,
                   n_shards: int = N_SHARDS_DEFAULT,
                   hands_per_shard: int = HANDS_PER_SHARD,
                   random_seed: int = RANDOM_SEED):

    os.makedirs(shards_dir, exist_ok=True)

    # Get unique hands and shuffle
    hands = pd.Index(pd.unique(df["hand_id"])).dropna()
    total_hands = len(hands)
    print(f"Total unique hands available: {total_hands:,}")

    if total_hands == 0:
        print("[shard] No hands to shard. Exiting.")
        return

    rng = np.random.default_rng(random_seed)
    hands = pd.Index(rng.permutation(hands.values))

    max_possible = total_hands // hands_per_shard
    n_actual = min(n_shards, max_possible)
    
    # Write each 
    for i in range(n_actual):
        start = i * hands_per_shard
        end   = start + hands_per_shard
        keep  = set(hands[start:end])
        shard_df = df[df["hand_id"].isin(keep)]

        shard_path = os.path.join(shards_dir, f"data(real)_{i+1:03d}.csv")
        shard_df.to_csv(shard_path, index=False)
        print(f"Wrote {shard_path}  (hands: {len(keep):,}, rows: {len(shard_df):,})")

    print(f"Wrote {n_actual} shard(s) to {shards_dir}")

def main():
    ap = argparse.ArgumentParser(description="Shard real poker data into N CSVs with K hands each.")
    ap.add_argument("--outdir", default=OUT_DIR_DEFAULT,
                    help="Folder containing parsed__*.csv or real_all_stakes.parquet (default: ./out)")
    ap.add_argument("--shards_dir", default=SHARDS_DIR_DEFAULT,
                    help="Folder to write Data (real)_N.csv (default: ./out/shards)")
    ap.add_argument("--n_shards", type=int, default=N_SHARDS_DEFAULT,
                    help="Number of CSV shards to produce (default: 500)")
    ap.add_argument("--hands_per_shard", type=int, default=HANDS_PER_SHARD,
                    help="Hands per shard (default: 1000)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    df = load_all_real_rows(args.outdir)

    # keep only rows with hole cards to ensure labeled decisions
    if {"hole1","hole2"}.issubset(df.columns):
        before = len(df)
        df = df[df["hole1"].notna() & df["hole2"].notna()]
        if len(df) != before:
            print(f"[clean] Kept labeled rows (with hole cards): {before:,} → {len(df):,}")

    shard_by_hands(df, args.shards_dir, args.n_shards, args.hands_per_shard, args.seed)

if __name__ == "__main__":
    main()
