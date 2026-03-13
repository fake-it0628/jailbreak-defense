# -*- coding: utf-8 -*-
"""
Risk Encoder 训练脚本
目的: 训练多轮风险记忆编码器
执行: python scripts/train_risk_encoder.py

Risk Encoder基于VAE框架，用于:
1. 学习多轮对话中的风险表示
2. 检测渐进式攻击和上下文稀释攻击
3. 累积跨轮次的风险信号
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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.risk_encoder import RiskEncoder


class MultiTurnDataset(Dataset):
    """多轮对话数据集"""
    
    def __init__(self, hidden_states_list, labels, max_turns=5):
        """
        Args:
            hidden_states_list: 列表，每个元素是一个对话的隐藏状态序列
            labels: 风险标签 (0/1)
            max_turns: 最大轮次数
        """
        self.hidden_states_list = hidden_states_list
        self.labels = labels
        self.max_turns = max_turns
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        hidden_seq = self.hidden_states_list[idx]
        label = self.labels[idx]
        
        # 填充或截断到max_turns
        if len(hidden_seq) > self.max_turns:
            hidden_seq = hidden_seq[:self.max_turns]
        elif len(hidden_seq) < self.max_turns:
            padding = torch.zeros(self.max_turns - len(hidden_seq), hidden_seq.shape[1])
            hidden_seq = torch.cat([hidden_seq, padding], dim=0)
        
        length = min(len(self.hidden_states_list[idx]), self.max_turns)
        
        return hidden_seq, torch.tensor(label, dtype=torch.float32), torch.tensor(length)


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


def extract_hidden_state(model, tokenizer, text, device, layer_idx):
    """提取单个文本的隐藏状态"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]
        seq_len = inputs["attention_mask"].sum() - 1
        return hidden_states[0, seq_len].cpu()


def generate_multi_turn_samples(model, tokenizer, device, layer_idx, num_samples=50):
    """
    生成多轮对话的模拟样本
    
    策略:
    1. 渐进式攻击: 从良性开始，逐渐升级到恶意
    2. 直接攻击: 第一轮就是恶意
    3. 良性对话: 全程良性
    """
    print("Generating multi-turn samples...")
    
    # 加载数据
    with open("data/processed/jailbreak_prompts.json", 'r', encoding='utf-8') as f:
        jailbreak_prompts = json.load(f)
    with open("data/processed/benign_prompts.json", 'r', encoding='utf-8') as f:
        benign_prompts = json.load(f)
    
    jb_texts = [p["prompt"] for p in jailbreak_prompts[:100]]
    bn_texts = [p["prompt"] for p in benign_prompts[:100]]
    
    samples = []
    labels = []
    
    # 1. 渐进式攻击 (标签=1)
    print("  Generating gradual escalation attacks...")
    for i in tqdm(range(min(num_samples // 3, len(jb_texts))), desc="Gradual"):
        # 构建: 良性 -> 良性 -> 恶意 -> 恶意
        turns = []
        for j in range(2):
            if j < len(bn_texts):
                h = extract_hidden_state(model, tokenizer, bn_texts[(i*2+j) % len(bn_texts)], device, layer_idx)
                turns.append(h)
        for j in range(2):
            if i+j < len(jb_texts):
                h = extract_hidden_state(model, tokenizer, jb_texts[(i+j) % len(jb_texts)], device, layer_idx)
                turns.append(h)
        
        if len(turns) >= 3:
            samples.append(torch.stack(turns))
            labels.append(1)  # 危险
    
    # 2. 直接攻击 (标签=1)
    print("  Generating direct attacks...")
    for i in tqdm(range(min(num_samples // 3, len(jb_texts))), desc="Direct"):
        turns = []
        for j in range(4):
            if i+j < len(jb_texts):
                h = extract_hidden_state(model, tokenizer, jb_texts[(i+j) % len(jb_texts)], device, layer_idx)
                turns.append(h)
        
        if len(turns) >= 2:
            samples.append(torch.stack(turns))
            labels.append(1)  # 危险
    
    # 3. 良性对话 (标签=0)
    print("  Generating benign conversations...")
    for i in tqdm(range(min(num_samples // 3, len(bn_texts))), desc="Benign"):
        turns = []
        for j in range(4):
            if i+j < len(bn_texts):
                h = extract_hidden_state(model, tokenizer, bn_texts[(i+j) % len(bn_texts)], device, layer_idx)
                turns.append(h)
        
        if len(turns) >= 2:
            samples.append(torch.stack(turns))
            labels.append(0)  # 安全
    
    return samples, labels


def train_epoch(model, dataloader, optimizer, device, kl_weight=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_kl_loss = 0
    correct = 0
    total = 0
    
    for hidden_seq, labels, lengths in dataloader:
        hidden_seq = hidden_seq.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(hidden_seq, labels, lengths, kl_weight)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += loss_dict["cls_loss"]
        total_kl_loss += loss_dict["kl_loss"]
        
        # 计算准确率
        with torch.no_grad():
            risk_scores = model.get_risk_score(hidden_seq, lengths)
            preds = (risk_scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return {
        "loss": total_loss / len(dataloader),
        "cls_loss": total_cls_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader),
        "accuracy": correct / total
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for hidden_seq, labels, lengths in dataloader:
            hidden_seq = hidden_seq.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            risk_scores = model.get_risk_score(hidden_seq, lengths)
            preds = (risk_scores > 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_scores.extend(risk_scores.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = correct / total
    
    # 计算TPR和FPR
    tp = sum(1 for s, l in zip(all_scores, all_labels) if s > 0.5 and l == 1)
    fp = sum(1 for s, l in zip(all_scores, all_labels) if s > 0.5 and l == 0)
    fn = sum(1 for s, l in zip(all_scores, all_labels) if s <= 0.5 and l == 1)
    tn = sum(1 for s, l in zip(all_scores, all_labels) if s <= 0.5 and l == 0)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr
    }


def main():
    parser = argparse.ArgumentParser(description="Train Risk Encoder")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" [TRAIN] Risk Encoder Training")
    print("=" * 60)
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    
    # 加载refusal direction获取layer信息
    direction_path = Path("checkpoints/steering/refusal_direction.pt")
    if direction_path.exists():
        direction_data = torch.load(direction_path, map_location="cpu", weights_only=False)
        layer_idx = direction_data["layer_idx"]
        hidden_dim = direction_data["direction_normalized"].shape[0]
    else:
        print("[WARN] Refusal direction not found, using defaults")
        layer_idx = -1
        hidden_dim = 896
    
    print(f"  Layer: {layer_idx}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # 加载LLM
    model, tokenizer = load_model(args.model_name, device)
    num_layers = model.config.num_hidden_layers
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx + 1
    
    # 生成多轮样本
    samples, labels = generate_multi_turn_samples(model, tokenizer, device, layer_idx, num_samples=60)
    
    print(f"\nTotal samples: {len(samples)}")
    print(f"  Dangerous: {sum(labels)}")
    print(f"  Safe: {len(labels) - sum(labels)}")
    
    # 划分训练/验证集
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    train_labels = labels[:split_idx]
    val_samples = samples[split_idx:]
    val_labels = labels[split_idx:]
    
    train_dataset = MultiTurnDataset(train_samples, train_labels)
    val_dataset = MultiTurnDataset(val_samples, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 创建Risk Encoder
    print(f"\nCreating Risk Encoder (latent_dim={args.latent_dim})...")
    risk_encoder = RiskEncoder(
        hidden_dim=hidden_dim,
        latent_dim=args.latent_dim,
        rnn_type="gru"
    )
    risk_encoder = risk_encoder.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(risk_encoder.parameters(), lr=args.lr)
    
    # 训练
    print(f"\nTraining for {args.epochs} epochs...")
    best_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(risk_encoder, train_loader, optimizer, device)
        val_metrics = evaluate(risk_encoder, val_loader, device)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            
            # 保存模型
            output_dir = Path("checkpoints/risk_encoder")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "model_state_dict": risk_encoder.state_dict(),
                "config": {
                    "hidden_dim": hidden_dim,
                    "latent_dim": args.latent_dim,
                    "layer_idx": layer_idx,
                    "model_name": args.model_name
                },
                "accuracy": best_acc,
                "epoch": epoch
            }, output_dir / "best_model.pt")
    
    # 最终评估
    print("\n" + "=" * 60)
    print(" Final Evaluation")
    print("=" * 60)
    
    final_metrics = evaluate(risk_encoder, val_loader, device)
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  TPR: {final_metrics['tpr']:.4f}")
    print(f"  FPR: {final_metrics['fpr']:.4f}")
    
    print(f"\n[OK] Best model saved to: checkpoints/risk_encoder/best_model.pt")
    print(f"     Best accuracy: {best_acc:.4f}")
    
    print("\n" + "=" * 60)
    print(" [DONE] Risk Encoder training complete!")
    print("=" * 60)
    print("\nNext step: python scripts/integrate_system.py")


if __name__ == "__main__":
    main()
