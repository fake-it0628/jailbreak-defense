# -*- coding: utf-8 -*-
"""
数据验证脚本
目的: 确认所有数据准备正确
执行: python scripts/verify_data.py
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path


def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")


def verify_raw_data():
    """验证原始数据"""
    print_header("原始数据验证")
    
    raw_dir = Path("data/raw")
    
    required = [
        ("advbench/advbench_prompts.json", "越狱提示(AdvBench种子)"),
        ("alpaca/alpaca_prompts.json", "良性指令"),
        ("jailbreak_templates/jailbreak_templates.json", "越狱模板"),
        ("training_pairs/refusal_samples.json", "拒绝样本"),
        ("training_pairs/compliance_samples.json", "遵从样本"),
        ("multi_turn/multi_turn_attacks.json", "多轮攻击"),
    ]
    optional = [
        ("harmful_expanded/harmful_unified.json", "合并有害集(v2, 运行 build_large_harmful_dataset.py 生成)"),
    ]
    
    all_ok = True
    for filename, desc in required:
        filepath = raw_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {desc}: {len(data)} 样本")
        else:
            print(f"✗ {desc}: 文件不存在 ({filepath})")
            all_ok = False

    for filename, desc in optional:
        filepath = raw_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {desc}: {len(data)} 样本")
        else:
            print(f"○ {desc}: 未生成（可选）→ {filepath}")
    
    return all_ok


def verify_processed_data():
    """验证处理后的数据"""
    print_header("处理后数据验证")
    
    processed_dir = Path("data/processed")
    
    files = [
        "jailbreak_prompts.json",
        "benign_prompts.json",
        "refusal_samples.json",
        "compliance_samples.json",
        "multi_turn_attacks.json"
    ]
    
    all_ok = True
    for filename in files:
        filepath = processed_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {filename}: {len(data)} 样本")
            
            # 显示示例
            if len(data) > 0 and isinstance(data[0], dict):
                print(f"    字段: {list(data[0].keys())}")
        else:
            print(f"✗ {filename}: 文件不存在")
            all_ok = False
    
    return all_ok


def verify_splits():
    """验证数据集划分"""
    print_header("数据集划分验证")
    
    splits_dir = Path("data/splits")
    
    for split in ["train", "val", "test"]:
        filepath = splits_dir / f"{split}.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {split}.json:")
            for key, value in data.items():
                print(f"    - {key}: {len(value)} 样本")
        else:
            print(f"✗ {split}.json: 文件不存在")


def show_samples():
    """显示样本示例"""
    print_header("样本示例")
    
    processed_dir = Path("data/processed")
    
    # 越狱样本
    jb_path = processed_dir / "jailbreak_prompts.json"
    if jb_path.exists():
        with open(jb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data:
            print(f"\n越狱样本示例:")
            print(f"  {data[0]['prompt'][:80]}...")
    
    # 良性样本
    bn_path = processed_dir / "benign_prompts.json"
    if bn_path.exists():
        with open(bn_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data:
            print(f"\n良性样本示例:")
            print(f"  {data[0]['prompt'][:80]}...")


def main():
    print("=" * 60)
    print(" 🔍 数据验证脚本")
    print("=" * 60)
    
    raw_ok = verify_raw_data()
    processed_ok = verify_processed_data()
    verify_splits()
    show_samples()
    
    print("\n" + "=" * 60)
    if raw_ok and processed_ok:
        print(" ✓ 数据验证通过!")
    else:
        print(" ⚠ 部分数据缺失，请检查")
    print("=" * 60)


if __name__ == "__main__":
    main()
