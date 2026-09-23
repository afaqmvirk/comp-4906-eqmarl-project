from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for path in (ROOT / "src", ROOT / "upstream" / "eqmarl"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
