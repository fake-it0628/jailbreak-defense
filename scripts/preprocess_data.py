# -*- coding: utf-8 -*-
"""
数据预处理脚本
目的: 将原始数据转换为训练所需格式
执行: python scripts/preprocess_data.py
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import random
from pathlib import Path

# 目录配置
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 随机种子
random.seed(42)


def load_json(path: Path):
    """加载JSON文件"""
    if not path.exists():
        print(f"⚠ 文件不存在: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path: Path):
    """保存JSON文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ 已保存: {path} ({len(data)} 样本)")


def process_jailbreak_prompts():
    """处理越狱攻击提示"""
    print("\n处理越狱攻击提示...")
    
    jailbreak_prompts = []
    
    # 加载AdvBench
    advbench_path = RAW_DIR / "advbench" / "advbench_prompts.json"
    if advbench_path.exists():
        advbench = load_json(advbench_path)
        for i, item in enumerate(advbench):
            jailbreak_prompts.append({
                "id": f"advbench_{i}",
                "prompt": item.get("prompt", ""),
                "category": item.get("category", "harmful_behavior"),
                "source": "advbench"
            })
    
    # 去重
    seen = set()
    unique = []
    for item in jailbreak_prompts:
        key = item["prompt"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    
    save_json(unique, PROCESSED_DIR / "jailbreak_prompts.json")
    return unique


def process_benign_prompts():
    """处理良性提示"""
    print("\n处理良性提示...")
    
    alpaca_path = RAW_DIR / "alpaca" / "alpaca_prompts.json"
    if not alpaca_path.exists():
        print("⚠ Alpaca数据不存在")
        return []
    
    alpaca = load_json(alpaca_path)
    
    benign_prompts = []
    for i, item in enumerate(alpaca):
        benign_prompts.append({
            "id": f"benign_{i}",
            "prompt": item.get("prompt", ""),
            "expected_output": item.get("output", ""),
            "category": "benign",
            "source": "alpaca"
        })
    
    save_json(benign_prompts, PROCESSED_DIR / "benign_prompts.json")
    return benign_prompts


def process_refusal_compliance():
    """处理拒绝/遵从样本"""
    print("\n处理拒绝/遵从样本...")
    
    refusal_path = RAW_DIR / "training_pairs" / "refusal_samples.json"
    compliance_path = RAW_DIR / "training_pairs" / "compliance_samples.json"
    
    refusal = load_json(refusal_path) if refusal_path.exists() else []
    compliance = load_json(compliance_path) if compliance_path.exists() else []
    
    # 添加ID
    for i, item in enumerate(refusal):
        item["id"] = f"refusal_{i}"
    for i, item in enumerate(compliance):
        item["id"] = f"compliance_{i}"
    
    save_json(refusal, PROCESSED_DIR / "refusal_samples.json")
    save_json(compliance, PROCESSED_DIR / "compliance_samples.json")
    
    return refusal, compliance


def process_multi_turn():
    """处理多轮攻击数据"""
    print("\n处理多轮攻击数据...")
    
    multi_turn_path = RAW_DIR / "multi_turn" / "multi_turn_attacks.json"
    if not multi_turn_path.exists():
        print("⚠ 多轮攻击数据不存在")
        return []
    
    multi_turn = load_json(multi_turn_path)
    save_json(multi_turn, PROCESSED_DIR / "multi_turn_attacks.json")
    return multi_turn


def create_splits():
    """创建训练/验证/测试集划分"""
    print("\n创建数据集划分...")
    
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载处理后的数据
    jailbreak = load_json(PROCESSED_DIR / "jailbreak_prompts.json")
    benign = load_json(PROCESSED_DIR / "benign_prompts.json")
    refusal = load_json(PROCESSED_DIR / "refusal_samples.json")
    compliance = load_json(PROCESSED_DIR / "compliance_samples.json")
    
    # 打乱
    random.shuffle(jailbreak)
    random.shuffle(benign)
    
    # 划分 70/15/15
    def split_data(data):
        n = len(data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        return {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
    
    jb_splits = split_data(jailbreak)
    bn_splits = split_data(benign)
    
    splits = {
        "train": {
            "jailbreak": jb_splits["train"],
            "benign": bn_splits["train"],
            "refusal": refusal,
            "compliance": compliance
        },
        "val": {
            "jailbreak": jb_splits["val"],
            "benign": bn_splits["val"]
        },
        "test": {
            "jailbreak": jb_splits["test"],
            "benign": bn_splits["test"]
        }
    }
    
    for split_name, split_data in splits.items():
        save_json(split_data, splits_dir / f"{split_name}.json")
    
    # 打印统计
    print("\n数据集划分统计:")
    print(f"  训练集: 越狱{len(jb_splits['train'])}, 良性{len(bn_splits['train'])}, 拒绝{len(refusal)}, 遵从{len(compliance)}")
    print(f"  验证集: 越狱{len(jb_splits['val'])}, 良性{len(bn_splits['val'])}")
    print(f"  测试集: 越狱{len(jb_splits['test'])}, 良性{len(bn_splits['test'])}")


def main():
    print("=" * 60)
    print(" 🔄 数据预处理脚本")
    print("=" * 60)
    
    # 处理各类数据
    jailbreak = process_jailbreak_prompts()
    benign = process_benign_prompts()
    refusal, compliance = process_refusal_compliance()
    multi_turn = process_multi_turn()
    
    # 创建数据集划分
    create_splits()
    
    print("\n" + "=" * 60)
    print(" ✓ 数据预处理完成!")
    print("=" * 60)
    print(f"\n处理后数据: data/processed/")
    print(f"数据集划分: data/splits/")
    print(f"\n下一步: python scripts/verify_data.py")


if __name__ == "__main__":
    main()
