# -*- coding: utf-8 -*-
"""
多模型规模实验脚本
目的: 在不同规模的LLM上测试HiSCaM防御系统，验证方法的泛化性

支持的模型:
- Qwen2.5-0.5B-Instruct (baseline, 已测试)
- Qwen2.5-1.5B-Instruct
- Qwen2.5-7B-Instruct
- Llama-2-7B-chat-hf
- Mistral-7B-Instruct-v0.2
- Llama-2-13B-chat-hf

执行: python scripts/multi_model_experiment.py --models qwen-0.5b,qwen-7b,llama-7b
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# 模型注册表
MODEL_REGISTRY = {
    "qwen-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "hidden_dim": 896,
        "num_layers": 24,
        "min_gpu_memory": 2,  # GB
    },
    "qwen-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "num_layers": 28,
        "min_gpu_memory": 4,
    },
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
        "num_layers": 28,
        "min_gpu_memory": 16,
    },
    "llama-7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "hidden_dim": 4096,
        "num_layers": 32,
        "min_gpu_memory": 16,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "hidden_dim": 4096,
        "num_layers": 32,
        "min_gpu_memory": 16,
    },
    "llama-13b": {
        "name": "meta-llama/Llama-2-13b-chat-hf",
        "hidden_dim": 5120,
        "num_layers": 40,
        "min_gpu_memory": 28,
    },
}


def get_available_gpu_memory() -> float:
    """获取可用GPU显存（GB）"""
    if not torch.cuda.is_available():
        return 0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except:
        return 0


def load_model_and_tokenizer(model_key: str, device: str = "auto"):
    """加载模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    config = MODEL_REGISTRY[model_key]
    model_name = config["name"]
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Expected hidden dim: {config['hidden_dim']}")
    print(f"Expected layers: {config['num_layers']}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 根据可用资源选择加载策略
    gpu_mem = get_available_gpu_memory()
    
    if device == "auto":
        if gpu_mem >= config["min_gpu_memory"]:
            device = "cuda"
        else:
            device = "cpu"
            print(f"[WARNING] GPU memory ({gpu_mem:.1f}GB) < required ({config['min_gpu_memory']}GB)")
            print("[WARNING] Falling back to CPU (will be slow)")
    
    load_kwargs = {
        "trust_remote_code": True,
        "output_hidden_states": True,
    }
    
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
        # 对于大模型启用量化
        if config["min_gpu_memory"] > 16 and gpu_mem < config["min_gpu_memory"]:
            print("[INFO] Enabling 8-bit quantization for large model")
            load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    actual_hidden_dim = model.config.hidden_size
    actual_num_layers = model.config.num_hidden_layers
    
    print(f"[OK] Model loaded successfully")
    print(f"  - Actual hidden dim: {actual_hidden_dim}")
    print(f"  - Actual num layers: {actual_num_layers}")
    print(f"  - Device: {next(model.parameters()).device}")
    
    return model, tokenizer, actual_hidden_dim, actual_num_layers


def extract_hidden_states(model, tokenizer, texts: List[str], layer_idx: int, device: str, max_length: int = 512):
    """提取指定层的隐藏状态"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer_idx]
    attention_mask = inputs["attention_mask"]
    
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    last_hidden = hidden_states[batch_indices, seq_lengths]
    
    return last_hidden


def train_prober_for_model(
    model,
    tokenizer,
    hidden_dim: int,
    num_layers: int,
    train_data: List[Dict],
    val_data: List[Dict],
    device: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3
) -> Tuple[object, Dict]:
    """为特定模型训练Safety Prober"""
    import torch.nn as nn
    import torch.nn.functional as F
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.safety_prober import SafetyProber
    
    layer_idx = num_layers  # 使用最后一层
    
    prober = SafetyProber(hidden_dim=hidden_dim, num_classes=2)
    prober = prober.to(device if device == "cpu" else model.device)
    
    optimizer = torch.optim.AdamW(prober.parameters(), lr=lr)
    
    best_acc = 0
    best_state = None
    training_history = []
    
    for epoch in range(1, epochs + 1):
        prober.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 打乱训练数据
        np.random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            texts = [item["prompt"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            
            if device == "cuda":
                labels = labels.to(model.device)
            else:
                labels = labels.to(device)
            
            hidden = extract_hidden_states(model, tokenizer, texts, layer_idx, device)
            
            optimizer.zero_grad()
            logits = prober(hidden)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(batch)
        
        train_acc = correct / total
        train_loss = total_loss / (len(train_data) // batch_size + 1)
        
        # 验证
        prober.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                texts = [item["prompt"] for item in batch]
                labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
                
                if device == "cuda":
                    labels = labels.to(model.device)
                else:
                    labels = labels.to(device)
                
                hidden = extract_hidden_states(model, tokenizer, texts, layer_idx, device)
                logits = prober(hidden)
                preds = logits.argmax(dim=-1)
                
                val_correct += (preds == labels).sum().item()
                val_total += len(batch)
                val_preds.extend(preds.cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())
        
        val_acc = val_correct / val_total
        
        # 计算TPR/FPR
        tp = sum(1 for p, l in zip(val_preds, val_labels_list) if p == 1 and l == 1)
        fn = sum(1 for p, l in zip(val_preds, val_labels_list) if p == 0 and l == 1)
        fp = sum(1 for p, l in zip(val_preds, val_labels_list) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(val_preds, val_labels_list) if p == 0 and l == 0)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_tpr": tpr,
            "val_fpr": fpr
        })
        
        print(f"  Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = prober.state_dict().copy()
    
    if best_state:
        prober.load_state_dict(best_state)
    
    return prober, {
        "best_val_acc": best_acc,
        "training_history": training_history,
        "layer_idx": layer_idx
    }


def evaluate_prober(
    prober,
    model,
    tokenizer,
    test_data: List[Dict],
    layer_idx: int,
    device: str,
    batch_size: int = 8
) -> Dict:
    """评估Safety Prober"""
    prober.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            texts = [item["prompt"] for item in batch]
            labels = [item["label"] for item in batch]
            
            hidden = extract_hidden_states(model, tokenizer, texts, layer_idx, device)
            logits = prober(hidden)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels)
            all_scores.extend(probs[:, 1].cpu().tolist())
    
    # 计算指标
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    asr = fn / (tp + fn) if (tp + fn) > 0 else 0  # Attack Success Rate
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # TPR
        "f1": f1,
        "fpr": fpr,
        "asr": asr,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    }


def load_data():
    """加载数据集"""
    train_data = []
    val_data = []
    test_data = []
    
    splits_dir = Path("data/splits")
    
    for split, data_list in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_path = splits_dir / f"{split}.json"
        if not split_path.exists():
            print(f"[WARNING] {split_path} not found")
            continue
        
        with open(split_path, 'r', encoding='utf-8') as f:
            split_json = json.load(f)
        
        for item in split_json.get("jailbreak", []):
            data_list.append({"prompt": item["prompt"], "label": 1, "type": "jailbreak"})
        
        for item in split_json.get("benign", []):
            data_list.append({"prompt": item["prompt"], "label": 0, "type": "benign"})
        
        for item in split_json.get("refusal", []):
            data_list.append({"prompt": item["prompt"], "label": 1, "type": "refusal"})
        
        for item in split_json.get("compliance", []):
            data_list.append({"prompt": item["prompt"], "label": 0, "type": "compliance"})
    
    return train_data, val_data, test_data


def run_experiment(model_key: str, train_data, val_data, test_data, device: str, output_dir: Path):
    """在单个模型上运行完整实验"""
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {model_key}")
    print(f"{'#'*70}")
    
    config = MODEL_REGISTRY[model_key]
    
    try:
        model, tokenizer, hidden_dim, num_layers = load_model_and_tokenizer(model_key, device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return {"model": model_key, "error": str(e)}
    
    # 训练
    print(f"\n[TRAIN] Training Safety Prober for {model_key}...")
    prober, train_info = train_prober_for_model(
        model, tokenizer, hidden_dim, num_layers,
        train_data, val_data, device
    )
    
    # 测试
    print(f"\n[TEST] Evaluating on test set...")
    test_metrics = evaluate_prober(
        prober, model, tokenizer, test_data,
        train_info["layer_idx"], device
    )
    
    # 保存结果
    result = {
        "model": model_key,
        "model_name": config["name"],
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "layer_used": train_info["layer_idx"],
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "best_val_acc": train_info["best_val_acc"],
        "test_metrics": test_metrics,
        "training_history": train_info["training_history"]
    }
    
    # 保存模型
    model_save_dir = output_dir / model_key
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": prober.state_dict(),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "layer_idx": train_info["layer_idx"],
        "model_key": model_key,
        "model_name": config["name"]
    }, model_save_dir / "safety_prober.pt")
    
    with open(model_save_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[RESULTS] {model_key}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall (TPR): {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  FPR: {test_metrics['fpr']:.4f}")
    print(f"  ASR: {test_metrics['asr']:.4f}")
    
    # 清理GPU内存
    del model
    del prober
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result


def generate_comparison_table(results: List[Dict], output_dir: Path):
    """生成模型对比表格"""
    print("\n" + "="*80)
    print(" MODEL COMPARISON RESULTS")
    print("="*80)
    
    # 表头
    header = f"{'Model':<20} {'Params':<10} {'Acc':<8} {'TPR':<8} {'FPR':<8} {'F1':<8} {'ASR':<8}"
    print(header)
    print("-" * 80)
    
    table_data = []
    for r in results:
        if "error" in r:
            print(f"{r['model']:<20} ERROR: {r['error']}")
            continue
        
        m = r["test_metrics"]
        params = f"{r['hidden_dim']/1000:.1f}k" if r['hidden_dim'] < 10000 else f"{r['hidden_dim']/1000:.0f}k"
        row = f"{r['model']:<20} {params:<10} {m['accuracy']:.4f}   {m['recall']:.4f}   {m['fpr']:.4f}   {m['f1']:.4f}   {m['asr']:.4f}"
        print(row)
        
        table_data.append({
            "model": r["model"],
            "model_name": r["model_name"],
            "hidden_dim": r["hidden_dim"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fpr": m["fpr"],
            "asr": m["asr"]
        })
    
    print("="*80)
    
    # 保存为JSON
    with open(output_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(table_data, f, indent=2)
    
    # 生成LaTeX表格
    latex_table = generate_latex_table(table_data)
    with open(output_dir / "comparison_table.tex", 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"\n[SAVED] Results saved to {output_dir}")


def generate_latex_table(table_data: List[Dict]) -> str:
    """生成LaTeX格式的对比表格"""
    latex = r"""
\begin{table}[h]
\centering
\caption{Performance Comparison Across Different Model Scales}
\label{tab:model_comparison}
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{Hidden Dim} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{FPR} & \textbf{ASR} \\
\midrule
"""
    
    for row in table_data:
        latex += f"{row['model']} & {row['hidden_dim']} & {row['accuracy']:.4f} & {row['precision']:.4f} & {row['recall']:.4f} & {row['f1']:.4f} & {row['fpr']:.4f} & {row['asr']:.4f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description="Multi-model scale experiment")
    parser.add_argument(
        "--models",
        type=str,
        default="qwen-0.5b",
        help="Comma-separated list of model keys (e.g., qwen-0.5b,qwen-7b,llama-7b)"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    print("="*70)
    print(" MULTI-MODEL SCALE EXPERIMENT")
    print(" HiSCaM: Hidden State Causal Monitoring for LLM Jailbreak Defense")
    print("="*70)
    
    # 解析模型列表
    model_keys = [m.strip() for m in args.models.split(",")]
    
    print(f"\nModels to test: {model_keys}")
    print(f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    # 验证模型
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            print(f"[ERROR] Unknown model: {key}")
            print(f"Available: {list(MODEL_REGISTRY.keys())}")
            return
    
    # GPU信息
    gpu_mem = get_available_gpu_memory()
    print(f"\nGPU Memory: {gpu_mem:.1f} GB" if gpu_mem > 0 else "\nNo GPU available, using CPU")
    
    # 加载数据
    print("\n[DATA] Loading datasets...")
    train_data, val_data, test_data = load_data()
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    if len(train_data) == 0:
        print("[ERROR] No training data found. Run data preparation first.")
        return
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/multi_model_experiment") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    all_results = []
    for model_key in model_keys:
        result = run_experiment(
            model_key, train_data, val_data, test_data,
            args.device, output_dir
        )
        all_results.append(result)
    
    # 生成对比报告
    generate_comparison_table(all_results, output_dir)
    
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
