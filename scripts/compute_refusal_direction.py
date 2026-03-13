# -*- coding: utf-8 -*-
"""
拒绝方向计算脚本
目的: 计算拒绝方向向量 (Refusal Direction)
执行: python scripts/compute_refusal_direction.py

方法: Difference-in-Means
refusal_direction = mean(refusal_hidden_states) - mean(compliance_hidden_states)

这个方向向量将用于Steering Matrix进行激活空间干预
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_name: str, device: str):
    """加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    return model, tokenizer


def extract_hidden_states(model, tokenizer, texts, device, layer_idx):
    """提取隐藏状态"""
    all_hidden = []
    
    for text in tqdm(texts, desc="Extracting hidden states"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if device != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            
            # 取最后一个token
            seq_len = inputs["attention_mask"].sum() - 1
            last_hidden = hidden_states[0, seq_len].cpu()
            all_hidden.append(last_hidden)
    
    return torch.stack(all_hidden)


def compute_refusal_direction(refusal_hidden, compliance_hidden):
    """
    计算拒绝方向
    
    方法: Difference-in-Means
    """
    refusal_mean = refusal_hidden.mean(dim=0)
    compliance_mean = compliance_hidden.mean(dim=0)
    
    direction = refusal_mean - compliance_mean
    
    # 归一化
    direction_normalized = direction / direction.norm()
    
    return direction, direction_normalized


def analyze_direction(refusal_hidden, compliance_hidden, direction):
    """分析拒绝方向的有效性"""
    # 计算各样本在方向上的投影
    refusal_proj = (refusal_hidden @ direction).numpy()
    compliance_proj = (compliance_hidden @ direction).numpy()
    
    # 计算分离度
    refusal_mean = refusal_proj.mean()
    compliance_mean = compliance_proj.mean()
    refusal_std = refusal_proj.std()
    compliance_std = compliance_proj.std()
    
    # Cohen's d (效应量)
    pooled_std = np.sqrt((refusal_std**2 + compliance_std**2) / 2)
    cohens_d = (refusal_mean - compliance_mean) / pooled_std if pooled_std > 0 else 0
    
    # 分类准确率 (使用0作为阈值)
    threshold = (refusal_mean + compliance_mean) / 2
    refusal_correct = (refusal_proj > threshold).sum()
    compliance_correct = (compliance_proj <= threshold).sum()
    accuracy = (refusal_correct + compliance_correct) / (len(refusal_proj) + len(compliance_proj))
    
    return {
        "refusal_mean_proj": float(refusal_mean),
        "compliance_mean_proj": float(compliance_mean),
        "separation": float(refusal_mean - compliance_mean),
        "cohens_d": float(cohens_d),
        "accuracy": float(accuracy),
        "threshold": float(threshold)
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Refusal Direction")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (-1 for last)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" [COMPUTE] Refusal Direction Calculation")
    print("=" * 60)
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    
    # 加载模型
    model, tokenizer = load_model(args.model_name, device)
    num_layers = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer + 1
    print(f"Using layer: {layer_idx}")
    
    # 加载拒绝/遵从样本
    print("\nLoading samples...")
    
    refusal_path = Path("data/processed/refusal_samples.json")
    compliance_path = Path("data/processed/compliance_samples.json")
    
    with open(refusal_path, 'r', encoding='utf-8') as f:
        refusal_samples = json.load(f)
    with open(compliance_path, 'r', encoding='utf-8') as f:
        compliance_samples = json.load(f)
    
    print(f"  Refusal samples: {len(refusal_samples)}")
    print(f"  Compliance samples: {len(compliance_samples)}")
    
    # 构建输入文本 (prompt + response)
    refusal_texts = [f"{s['prompt']}\n\n{s['response']}" for s in refusal_samples]
    compliance_texts = [f"{s['prompt']}\n\n{s['response']}" for s in compliance_samples]
    
    # 提取隐藏状态
    print("\nExtracting hidden states...")
    refusal_hidden = extract_hidden_states(model, tokenizer, refusal_texts, device, layer_idx)
    compliance_hidden = extract_hidden_states(model, tokenizer, compliance_texts, device, layer_idx)
    
    print(f"  Refusal hidden shape: {refusal_hidden.shape}")
    print(f"  Compliance hidden shape: {compliance_hidden.shape}")
    
    # 计算拒绝方向
    print("\nComputing refusal direction...")
    direction, direction_normalized = compute_refusal_direction(refusal_hidden, compliance_hidden)
    
    print(f"  Direction norm: {direction.norm():.4f}")
    print(f"  Normalized direction norm: {direction_normalized.norm():.4f}")
    
    # 分析方向有效性
    print("\nAnalyzing direction effectiveness...")
    analysis = analyze_direction(refusal_hidden, compliance_hidden, direction_normalized)
    
    print(f"  Refusal mean projection: {analysis['refusal_mean_proj']:.4f}")
    print(f"  Compliance mean projection: {analysis['compliance_mean_proj']:.4f}")
    print(f"  Separation: {analysis['separation']:.4f}")
    print(f"  Cohen's d: {analysis['cohens_d']:.4f}")
    print(f"  Classification accuracy: {analysis['accuracy']:.4f}")
    
    # 保存结果
    output_dir = Path("checkpoints/steering")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存方向向量
    torch.save({
        "direction": direction,
        "direction_normalized": direction_normalized,
        "layer_idx": layer_idx,
        "model_name": args.model_name,
        "analysis": analysis,
        "num_refusal_samples": len(refusal_samples),
        "num_compliance_samples": len(compliance_samples)
    }, output_dir / "refusal_direction.pt")
    
    print(f"\n[OK] Saved to: {output_dir / 'refusal_direction.pt'}")
    
    # 保存分析结果
    with open(output_dir / "refusal_direction_analysis.json", 'w') as f:
        json.dump({
            "model_name": args.model_name,
            "layer_idx": layer_idx,
            "direction_norm": float(direction.norm()),
            **analysis
        }, f, indent=2)
    
    print(f"[OK] Saved analysis to: {output_dir / 'refusal_direction_analysis.json'}")
    
    # 判断效果
    print("\n" + "=" * 60)
    if analysis['cohens_d'] > 0.8:
        print(" [GOOD] Strong separation (Cohen's d > 0.8)")
    elif analysis['cohens_d'] > 0.5:
        print(" [OK] Medium separation (Cohen's d > 0.5)")
    else:
        print(" [WARN] Weak separation (Cohen's d < 0.5)")
    print("=" * 60)
    
    print("\nNext step: python scripts/train_steering_matrix.py")


if __name__ == "__main__":
    main()
