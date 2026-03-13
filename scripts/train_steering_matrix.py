# -*- coding: utf-8 -*-
"""
Steering Matrix 训练脚本
目的: 训练激活空间干预矩阵
执行: python scripts/train_steering_matrix.py

Steering Matrix用于在检测到恶意输入时，
将隐藏状态"转向"到会产生拒绝响应的方向
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.steering_matrix import SteeringMatrix


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


def extract_hidden_states_batch(model, tokenizer, texts, device, layer_idx, batch_size=4):
    """批量提取隐藏状态"""
    all_hidden = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=512)
        if device != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            
            # 取每个样本最后一个token
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths].cpu()
            all_hidden.append(last_hidden)
    
    return torch.cat(all_hidden, dim=0)


def train_steering_matrix(
    steering_matrix,
    refusal_direction,
    jailbreak_hidden,
    benign_hidden,
    device,
    epochs=50,
    lr=1e-3
):
    """训练Steering Matrix"""
    steering_matrix.train()
    optimizer = torch.optim.Adam(steering_matrix.parameters(), lr=lr)
    
    # 将数据移到设备
    jailbreak_hidden = jailbreak_hidden.to(device)
    benign_hidden = benign_hidden.to(device)
    refusal_direction = refusal_direction.to(device)
    
    # 归一化拒绝方向
    refusal_dir_norm = F.normalize(refusal_direction, dim=0)
    
    history = {"loss": [], "jailbreak_shift": [], "benign_shift": []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 对恶意样本进行干预
        jb_steered = steering_matrix(jailbreak_hidden, steering_strength=1.0)
        
        # 对良性样本进行干预（应该影响最小）
        bn_steered = steering_matrix(benign_hidden, steering_strength=1.0)
        
        # 计算在拒绝方向上的移动
        jb_shift = ((jb_steered - jailbreak_hidden) @ refusal_dir_norm).mean()
        bn_shift = ((bn_steered - benign_hidden) @ refusal_dir_norm).mean()
        
        # 损失函数:
        # 1. 恶意样本应该向拒绝方向移动 (最大化 jb_shift)
        # 2. 良性样本应该保持不变 (最小化 bn_shift)
        # 3. 良性样本的总体变化应该最小 (零空间约束)
        
        loss_jb = -jb_shift  # 负号因为要最大化
        loss_bn = bn_shift.abs()  # 最小化良性样本的移动
        loss_preserve = ((bn_steered - benign_hidden) ** 2).mean()  # 保持良性不变
        
        loss = loss_jb + 2.0 * loss_bn + 0.5 * loss_preserve
        
        loss.backward()
        optimizer.step()
        
        history["loss"].append(loss.item())
        history["jailbreak_shift"].append(jb_shift.item())
        history["benign_shift"].append(bn_shift.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, "
                  f"JB_shift={jb_shift.item():.4f}, BN_shift={bn_shift.item():.4f}")
    
    return history


def evaluate_steering(steering_matrix, refusal_direction, jailbreak_hidden, benign_hidden, device):
    """评估Steering效果"""
    steering_matrix.eval()
    
    jailbreak_hidden = jailbreak_hidden.to(device)
    benign_hidden = benign_hidden.to(device)
    refusal_direction = refusal_direction.to(device)
    
    refusal_dir_norm = F.normalize(refusal_direction, dim=0)
    
    with torch.no_grad():
        # 干预
        jb_steered = steering_matrix(jailbreak_hidden, steering_strength=1.0)
        bn_steered = steering_matrix(benign_hidden, steering_strength=1.0)
        
        # 计算移动
        jb_shift = ((jb_steered - jailbreak_hidden) @ refusal_dir_norm)
        bn_shift = ((bn_steered - benign_hidden) @ refusal_dir_norm)
        
        # 计算干预前后在拒绝方向上的投影
        jb_proj_before = (jailbreak_hidden @ refusal_dir_norm)
        jb_proj_after = (jb_steered @ refusal_dir_norm)
        
        bn_proj_before = (benign_hidden @ refusal_dir_norm)
        bn_proj_after = (bn_steered @ refusal_dir_norm)
        
        # 计算良性样本的变化幅度
        bn_change_norm = (bn_steered - benign_hidden).norm(dim=1)
        jb_change_norm = (jb_steered - jailbreak_hidden).norm(dim=1)
    
    return {
        "jailbreak_shift_mean": jb_shift.mean().item(),
        "jailbreak_shift_std": jb_shift.std().item(),
        "benign_shift_mean": bn_shift.mean().item(),
        "benign_shift_std": bn_shift.std().item(),
        "jailbreak_proj_before": jb_proj_before.mean().item(),
        "jailbreak_proj_after": jb_proj_after.mean().item(),
        "benign_proj_before": bn_proj_before.mean().item(),
        "benign_proj_after": bn_proj_after.mean().item(),
        "benign_change_norm_mean": bn_change_norm.mean().item(),
        "jailbreak_change_norm_mean": jb_change_norm.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Steering Matrix")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" [TRAIN] Steering Matrix Training")
    print("=" * 60)
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    
    # 加载拒绝方向
    print("\nLoading refusal direction...")
    direction_path = Path("checkpoints/steering/refusal_direction.pt")
    if not direction_path.exists():
        print("[ERROR] Refusal direction not found. Run compute_refusal_direction.py first.")
        return
    
    direction_data = torch.load(direction_path, map_location="cpu", weights_only=False)
    refusal_direction = direction_data["direction_normalized"]
    layer_idx = direction_data["layer_idx"]
    hidden_dim = refusal_direction.shape[0]
    
    print(f"  Refusal direction loaded (dim={hidden_dim})")
    print(f"  Layer: {layer_idx}")
    
    # 加载模型
    model, tokenizer = load_model(args.model_name, device)
    
    # 加载训练数据
    print("\nLoading training data...")
    
    with open("data/splits/train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    jailbreak_texts = [s["prompt"] for s in train_data.get("jailbreak", [])][:100]
    benign_texts = [s["prompt"] for s in train_data.get("benign", [])][:100]
    
    print(f"  Jailbreak samples: {len(jailbreak_texts)}")
    print(f"  Benign samples: {len(benign_texts)}")
    
    # 提取隐藏状态
    print("\nExtracting hidden states...")
    jailbreak_hidden = extract_hidden_states_batch(model, tokenizer, jailbreak_texts, device, layer_idx)
    benign_hidden = extract_hidden_states_batch(model, tokenizer, benign_texts, device, layer_idx)
    
    print(f"  Jailbreak hidden: {jailbreak_hidden.shape}")
    print(f"  Benign hidden: {benign_hidden.shape}")
    
    # 创建Steering Matrix
    print(f"\nCreating Steering Matrix (rank={args.rank})...")
    steering_matrix = SteeringMatrix(
        hidden_dim=hidden_dim,
        rank=args.rank,
        use_null_space=True
    )
    
    # 设置拒绝方向
    steering_matrix.refusal_direction.data = refusal_direction.clone()
    
    # 更新零空间（基于良性样本）
    print("Computing null space projection...")
    steering_matrix.update_null_space(benign_hidden, threshold=0.95)
    
    steering_matrix = steering_matrix.to(device)
    
    # 训练
    print(f"\nTraining for {args.epochs} epochs...")
    history = train_steering_matrix(
        steering_matrix,
        refusal_direction,
        jailbreak_hidden,
        benign_hidden,
        device,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # 评估
    print("\nEvaluating...")
    eval_results = evaluate_steering(
        steering_matrix, refusal_direction, jailbreak_hidden, benign_hidden, device
    )
    
    print("\nEvaluation Results:")
    print(f"  Jailbreak shift (mean): {eval_results['jailbreak_shift_mean']:.4f}")
    print(f"  Benign shift (mean): {eval_results['benign_shift_mean']:.4f}")
    print(f"  Jailbreak projection: {eval_results['jailbreak_proj_before']:.4f} -> {eval_results['jailbreak_proj_after']:.4f}")
    print(f"  Benign projection: {eval_results['benign_proj_before']:.4f} -> {eval_results['benign_proj_after']:.4f}")
    print(f"  Benign change norm: {eval_results['benign_change_norm_mean']:.4f}")
    print(f"  Jailbreak change norm: {eval_results['jailbreak_change_norm_mean']:.4f}")
    
    # 保存模型
    output_dir = Path("checkpoints/steering")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": steering_matrix.state_dict(),
        "config": {
            "hidden_dim": hidden_dim,
            "rank": args.rank,
            "layer_idx": layer_idx,
            "model_name": args.model_name
        },
        "history": history,
        "eval_results": eval_results
    }, output_dir / "steering_matrix.pt")
    
    print(f"\n[OK] Saved to: {output_dir / 'steering_matrix.pt'}")
    
    # 保存评估结果
    with open(output_dir / "steering_eval.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"[OK] Saved evaluation to: {output_dir / 'steering_eval.json'}")
    
    print("\n" + "=" * 60)
    print(" [DONE] Steering Matrix training complete!")
    print("=" * 60)
    print("\nNext step: python scripts/train_risk_encoder.py")


if __name__ == "__main__":
    main()
