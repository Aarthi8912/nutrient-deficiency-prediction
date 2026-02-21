"""
Enhanced pipeline: 14 inputs (demographics, clinical, physiological, diet).
Output: data/nfhs5_enhanced.csv with real BMI (kg/m²). Requires NFHS5_IR.csv.
"""
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
input_file = "NFHS5_IR.csv"
output_file = "data/nfhs5_enhanced.csv"

# Columns to extract
cols = {
    # Original
    "v012": "age",
    "v445": "bmi",      # Needs /100
    "v457": "anemia",   # 1-4
    "v106": "education",# 0-3
    "v190": "wealth",   # 1-5
    "v025": "residence",# 1=Urban, 2=Rural

    # Physiological
    "v213": "pregnant",      # 0=No, 1=Yes
    "v404": "breastfeeding", # 0=No, 1=Yes
    "v481": "insurance",     # 0=No, 1=Yes

    # Diet (frequency or Yes/No - usually 0=No, 1=Yes in Recode)
    # Checking raw data: 0=No, 1=Yes usually.
    "v414n": "milk",
    "v414e": "eggs",
    "v414p": "fruit",
    "v414s": "pulses", # Assuming s/v exists, taking common ones
    "v414v": "curd"
}

chunksize = 50000
saved_rows = 0
first_chunk = True

print("🚀 Starting Enhanced Data Extraction...")

for chunk in pd.read_csv(input_file, chunksize=chunksize, usecols=list(cols.keys()), low_memory=False):
    
    # 1. Drop rows where Diet is missing (The "Premium" filter)
    # We use 'v414n' (Milk) as proxy for diet module participation
    chunk = chunk.dropna(subset=["v414n"])
    
    if chunk.empty:
        processed_rows += chunksize
        continue

    # 2. Rename
    chunk = chunk.rename(columns=cols)

    # 3. BMI: NFHS decimal*100 → real kg/m²
    chunk["bmi"] = pd.to_numeric(chunk["bmi"], errors="coerce")
    chunk = chunk[(chunk["bmi"] >= 100) & (chunk["bmi"] < 6000)]
    chunk["bmi"] = chunk["bmi"] / 100.0

    # 4. Clean Anemia (9=Missing)
    chunk = chunk[chunk["anemia"] < 9]

    # 5. Binary vars → 0/1
    for c in ["pregnant", "breastfeeding", "insurance", "milk", "eggs", "fruit", "pulses", "curd"]:
        if c in chunk.columns:
            chunk[c] = chunk[c].fillna(0).astype(int)
            chunk[c] = chunk[c].apply(lambda x: 1 if x == 1 else 0)

    # 6. Risk = underweight (BMI < 18.5) OR severe/moderate anemia (1 or 2)
    chunk["nutrition_risk"] = ((chunk["bmi"] < 18.5) | (chunk["anemia"].isin([1, 2]))).astype(int)

    # 7. Save
    if not chunk.empty:
        chunk.to_csv(output_file, mode="w" if first_chunk else "a", index=False, header=first_chunk)
        first_chunk = False
        saved_rows += len(chunk)

    print(f"   Saved {saved_rows} rows...")

print(f"✅ Done! Saved {saved_rows} high-quality rows to {output_file}")
