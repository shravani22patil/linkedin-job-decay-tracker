import pandas as pd
from pathlib import Path

raw = Path("data/raw")
out = raw / "demo"

out.mkdir(exist_ok=True)

files = [
    "LinkedIn_April.csv",
    "LinkedIn_May_2026.csv",
    "June_2026.csv"
]

for f in files:
    df = pd.read_csv(raw / f, encoding="latin1")

    # keep first 3000 rows
    df = df.head(3000)

    df.to_csv(out / f, index=False)

print("Demo dataset created.")