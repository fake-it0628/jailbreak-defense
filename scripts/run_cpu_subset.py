# -*- coding: utf-8 -*-
"""
CPU 子集流水线（限量样本 + 越狱侧优先文言文）

用法:
  python scripts/run_cpu_subset.py --dry-run          # 仅统计子集规模（输出到终端，勿提交日志）
  python scripts/run_cpu_subset.py --device cpu       # 按默认限额跑 train -> refusal -> steering -> risk -> integrate -> eval

默认限额可按机器调整；评测结果写入 results/（已被 .gitignore 忽略）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_subset import apply_category_limits, count_wenyan_jailbreak


def _load_split(name: str) -> dict:
    p = ROOT / "data" / "splits" / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"缺少划分文件: {p}（请先 preprocess_data）")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def print_stats(
    train_jb: int,
    train_bn: int,
    val_jb: int,
    val_bn: int,
    test_jb: int,
    test_bn: int,
    prefer: bool,
) -> None:
    for label, raw, jb_lim, bn_lim in (
        ("train", _load_split("train"), train_jb, train_bn),
        ("val", _load_split("val"), val_jb, val_bn),
        ("test", _load_split("test"), test_jb, test_bn),
    ):
        sub = apply_category_limits(
            raw,
            jailbreak_limit=jb_lim,
            benign_limit=bn_lim,
            prefer_wenyan=prefer,
        )
        jb = sub.get("jailbreak", [])
        bn = sub.get("benign", [])
        wy = count_wenyan_jailbreak(jb)
        print(
            f"  [{label}] jailbreak={len(jb)} (wenyan_cc_bos_style={wy}), benign={len(bn)}, "
            f"prefer_wenyan={prefer}"
        )


def run_cmd(args_list: list[str]) -> None:
    print("\n>>", " ".join(args_list))
    r = subprocess.run(args_list, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train_jailbreak_limit", type=int, default=480)
    ap.add_argument("--train_benign_limit", type=int, default=480)
    ap.add_argument("--val_jailbreak_limit", type=int, default=120)
    ap.add_argument("--val_benign_limit", type=int, default=120)
    ap.add_argument("--eval_jailbreak_limit", type=int, default=200)
    ap.add_argument("--eval_benign_limit", type=int, default=200)
    ap.add_argument("--prefer_wenyan", dest="prefer_wenyan", action="store_true", default=True)
    ap.add_argument("--no_prefer_wenyan", dest="prefer_wenyan", action="store_false")
    ap.add_argument("--prober_epochs", type=int, default=3)
    ap.add_argument("--prober_batch", type=int, default=4)
    ap.add_argument("--steering_epochs", type=int, default=10)
    ap.add_argument("--risk_epochs", type=int, default=10)
    args = ap.parse_args()
    prefer = args.prefer_wenyan

    print("=" * 60)
    print(" CPU subset pipeline (HiSCaM)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] 子集规模（终端输出请勿当作文档提交）:\n")
        print_stats(
            args.train_jailbreak_limit,
            args.train_benign_limit,
            args.val_jailbreak_limit,
            args.val_benign_limit,
            args.eval_jailbreak_limit,
            args.eval_benign_limit,
            prefer,
        )
        print(
            "\n说明: train/val/test 行对应 prober 训练用划分；"
            "evaluate_benchmark 使用与 test 相同的限量。"
        )
        print("\n[DRY-RUN] 结束。")
        return

    py = sys.executable
    train_args = [
        f"--jailbreak_limit={args.train_jailbreak_limit}",
        f"--benign_limit={args.train_benign_limit}",
        f"--val_jailbreak_limit={args.val_jailbreak_limit}",
        f"--val_benign_limit={args.val_benign_limit}",
        f"--test_jailbreak_limit={args.eval_jailbreak_limit}",
        f"--test_benign_limit={args.eval_benign_limit}",
    ]
    if prefer:
        train_args.append("--prefer_wenyan")

    run_cmd(
        [
            py,
            "scripts/train_safety_prober.py",
            "--device",
            args.device,
            "--model_name",
            args.model_name,
            "--epochs",
            str(args.prober_epochs),
            "--batch_size",
            str(args.prober_batch),
        ]
        + train_args
    )

    run_cmd(
        [
            py,
            "scripts/compute_refusal_direction.py",
            "--device",
            args.device,
            "--model_name",
            args.model_name,
        ]
    )

    run_cmd(
        [
            py,
            "scripts/train_steering_matrix.py",
            "--device",
            args.device,
            "--model_name",
            args.model_name,
            "--epochs",
            str(args.steering_epochs),
        ]
    )

    run_cmd(
        [
            py,
            "scripts/train_risk_encoder.py",
            "--device",
            args.device,
            "--model_name",
            args.model_name,
            "--epochs",
            str(args.risk_epochs),
        ]
    )

    run_cmd([py, "scripts/integrate_system.py"])

    eval_cmd = [
        py,
        "scripts/evaluate_benchmark.py",
        "--device",
        args.device,
        "--model_name",
        args.model_name,
        "--jailbreak_limit",
        str(args.eval_jailbreak_limit),
        "--benign_limit",
        str(args.eval_benign_limit),
    ]
    if prefer:
        eval_cmd.append("--prefer_wenyan")
    run_cmd(eval_cmd)

    print("\n[OK] CPU 子集流水线结束。metrics 在 results/（默认不进入 git）。")


if __name__ == "__main__":
    main()
