import os
import sys
import json
import random
from pathlib import Path
from typing import List


def load_json_list(path: Path) -> List[dict]:
    """Load a JSON file expected to be a list (WebQSP / GrailQA format)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at {path}, got {type(data)}")
    return data


def sample_list(items: List[dict], n: int) -> List[dict]:
    n = min(n, len(items))
    return random.sample(items, n) if n > 0 else []


def write_output(sampled: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)


def process_one(src: Path, dst: Path, n: int) -> None:
    data = load_json_list(src)
    sampled = sample_list(data, n)
    write_output(sampled, dst)
    print(f"Sampled {len(sampled)} items from {src} -> {dst}")
    print(f"Sampled {len(sampled)} items from {src} -> {dst}")

def main():
    seed = os.environ.get("SAMPLE_SEED", "42")
    seed = os.environ.get("SAMPLE_SEED", "42")
    if seed is not None:
        try:
            random.seed(int(seed))
        except ValueError:
            random.seed(seed)

    # 以本脚本所在 data/scripts 为基准，定位到 data 目录
    data_root = Path(__file__).resolve().parents[1]

    # 路径
    webqsp_in = data_root / "webqsp" / "webqsp.json"
    webqsp_out = data_root / "webqsp" / "webqsp_500.json"
    grailqa_in = data_root / "grailqa" / "grailqa.json"
    grailqa_out = data_root / "grailqa" / "grailqa_100.json"

    # 基础检查
    for p in [webqsp_in, grailqa_in]:
        if not p.exists():
            print(f"Error: input not found: {p}", file=sys.stderr)
            sys.exit(1)
    # 采样各100条
    # 采样各100条
    process_one(webqsp_in, webqsp_out, 500)
    # process_one(grailqa_in, grailqa_out, 100)

if __name__ == "__main__":
    main()
