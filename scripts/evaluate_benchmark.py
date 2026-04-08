# -*- coding: utf-8 -*-
"""
基准评估脚本
目的: 在测试集上评估防御系统性能
执行: python scripts/evaluate_benchmark.py

评估指标:
- ASR (Attack Success Rate): 越狱攻击成功率 (越低越好)
- FPR (False Positive Rate): 误报率 (越低越好)
- TPR (True Positive Rate): 真正例率 / 召回率 (越高越好)
- Inference Latency: 推理延迟
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.defense_system import JailbreakDefenseSystem
from src.data_subset import apply_category_limits


def load_test_data(test_path: Path):
    """加载测试数据"""
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_on_dataset(
    defense_system,
    llm_model,
    tokenizer,
    data,
    category: str,
    device: str,
    num_layers: int
):
    """在单个数据集上评估"""
    samples = data.get(category, [])
    if not samples:
        return None
    
    is_harmful_category = category in ["jailbreak", "refusal"]
    
    results = {
        "total": len(samples),
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "latencies": []
    }
    
    for item in tqdm(samples, desc=f"Evaluating {category}"):
        prompt = item.get("prompt", "")
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if device != "cpu":
            inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
        
        # 计时
        start_time = time.time()
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[num_layers]
            seq_len = inputs["attention_mask"].sum() - 1
            last_hidden = hidden_states[0, seq_len].unsqueeze(0)
            
            # 防御检测
            result = defense_system(last_hidden)
        
        latency = time.time() - start_time
        results["latencies"].append(latency)
        
        # 统计
        detected_harmful = result.is_harmful
        
        if is_harmful_category:
            if detected_harmful:
                results["true_positive"] += 1
            else:
                results["false_negative"] += 1
        else:
            if detected_harmful:
                results["false_positive"] += 1
            else:
                results["true_negative"] += 1
    
    return results


def compute_metrics(results_dict):
    """计算综合指标"""
    tp = sum(r.get("true_positive", 0) for r in results_dict.values() if r)
    fp = sum(r.get("false_positive", 0) for r in results_dict.values() if r)
    tn = sum(r.get("true_negative", 0) for r in results_dict.values() if r)
    fn = sum(r.get("false_negative", 0) for r in results_dict.values() if r)
    
    all_latencies = []
    for r in results_dict.values():
        if r:
            all_latencies.extend(r.get("latencies", []))
    
    # 计算指标
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
    
    # ASR = 攻击成功率 = 漏检的有害样本比例
    asr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "asr": asr,  # Attack Success Rate (越低越好)
        "avg_latency_ms": avg_latency * 1000,
        "confusion_matrix": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Defense System")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--jailbreak_limit", type=int, default=0, help="测试集越狱上限（0 不限制）")
    parser.add_argument("--benign_limit", type=int, default=0, help="测试集良性上限（0 不限制）")
    parser.add_argument(
        "--prefer_wenyan",
        action="store_true",
        help="越狱限量时优先保留 wenyan_cc_bos_style",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(" [EVAL] Benchmark Evaluation")
    print("=" * 60)
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    
    # 加载模型
    print(f"\nLoading model: {args.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        llm_model = llm_model.to(device)
    llm_model.eval()
    
    hidden_dim = llm_model.config.hidden_size
    num_layers = llm_model.config.num_hidden_layers
    
    # 创建防御系统
    print("\nCreating defense system...")
    defense_system = JailbreakDefenseSystem(hidden_dim=hidden_dim)
    
    # 加载训练好的权重
    prober_path = Path("checkpoints/safety_prober/best_model.pt")
    if prober_path.exists():
        checkpoint = torch.load(prober_path, map_location=device)
        defense_system.safety_prober.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Loaded Safety Prober")
    else:
        print("[WARN] Using random weights")
    
    defense_system = defense_system.to(device)
    defense_system.eval()
    
    # 加载测试数据
    test_path = Path("data/splits/test.json")
    if not test_path.exists():
        print(f"[ERROR] Test data not found: {test_path}")
        return
    
    test_data = load_test_data(test_path)
    test_data = apply_category_limits(
        test_data,
        jailbreak_limit=args.jailbreak_limit,
        benign_limit=args.benign_limit,
        prefer_wenyan=args.prefer_wenyan,
    )
    if args.jailbreak_limit > 0 or args.benign_limit > 0:
        print(
            f"[subset] test jailbreak={len(test_data.get('jailbreak', []))}, "
            f"benign={len(test_data.get('benign', []))}"
        )

    # 评估各类别
    print("\nEvaluating...")
    results_dict = {}
    
    for category in ["jailbreak", "benign"]:
        results = evaluate_on_dataset(
            defense_system, llm_model, tokenizer,
            test_data, category, device, num_layers
        )
        results_dict[category] = results
    
    # 计算指标
    metrics = compute_metrics(results_dict)
    
    # 输出结果
    print("\n" + "=" * 60)
    print(" [RESULTS] Evaluation Results")
    print("=" * 60)
    
    print(f"\n  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"\n  TPR (Detection Rate): {metrics['tpr']:.4f}")
    print(f"  FPR (False Alarm):    {metrics['fpr']:.4f}")
    print(f"  ASR (Attack Success): {metrics['asr']:.4f}")
    print(f"\n  Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
    
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TP: {cm['TP']:4d}  |  FP: {cm['FP']:4d}")
    print(f"    FN: {cm['FN']:4d}  |  TN: {cm['TN']:4d}")
    
    # 保存结果
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
