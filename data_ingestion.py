from pathlib import Path
import pandas as pd

BASE_DIR    = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE  = BASE_DIR / "train.csv"
INPUT_FILE2 = BASE_DIR / "test.csv"
OUTPUT_FILE = INGESTED_DIR / "train.csv"
OUTPUT_FILE2 = INGESTED_DIR / "test.csv"

def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    df2 = pd.read_csv(INPUT_FILE2)
    assert not df.empty, "Dataset is empty"
    df.to_csv(OUTPUT_FILE, index=False)
    df2.to_csv(OUTPUT_FILE2, index=False)
    print(f"Train ingested: {INPUT_FILE} → {OUTPUT_FILE}")
    print(f"Test ingested: {INPUT_FILE2} → {OUTPUT_FILE2}")

if __name__ == "__main__":
    ingest_data()