#!/usr/bin/env bash
set -e
echo "=== Preparing DiverseVul dataset ==="

# Download DiverseVul (requires Hugging Face access)
python - <<'PYTHON'
from datasets import load_dataset
import json, pathlib

ds = load_dataset("wagner-lab/diversevul", split="train")
out = pathlib.Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

samples = [{"func": row["func"], "target": row["target"], "cwe": row.get("cwe", "")} for row in ds]
with open(out / "diversevul_train.json", "w") as f:
    json.dump(samples, f)
print(f"Saved {len(samples)} samples to data/processed/diversevul_train.json")
PYTHON

echo "=== Partitioning (10 clients, Dirichlet alpha=0.5) ==="
python src/utils/partition.py --n_clients 10 --alpha 0.5
echo "Done."
