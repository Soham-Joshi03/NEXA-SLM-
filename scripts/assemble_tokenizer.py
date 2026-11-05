from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "tokenizer"
PARTS = [
    PARTS_DIR / "jamba_bbpe_tokenizer.json.part-01",
    PARTS_DIR / "jamba_bbpe_tokenizer.json.part-02",
]
OUTPUT = ROOT / "jamba_bbpe_tokenizer.json"

missing = [str(path) for path in PARTS if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing tokenizer part(s): " + ", ".join(missing))

# The parts are consecutive text ranges from the original JSON file.
OUTPUT.write_text("".join(path.read_text(encoding="utf-8") for path in PARTS), encoding="utf-8")

# Validate the reconstructed tokenizer JSON before using it.
with OUTPUT.open("r", encoding="utf-8") as handle:
    tokenizer = json.load(handle)

vocab_size = len(tokenizer["model"]["vocab"])
merge_count = len(tokenizer["model"].get("merges", []))
print(f"Tokenizer assembled and validated: {OUTPUT}")
print(f"Vocabulary entries: {vocab_size:,}")
print(f"BPE merges: {merge_count:,}")
