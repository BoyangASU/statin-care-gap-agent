import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    return p
