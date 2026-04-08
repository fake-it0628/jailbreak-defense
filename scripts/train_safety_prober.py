# -*- coding: utf-8 -*-
"""
Safety Prober 训练脚本
目的: 训练一个探测器来区分有害/无害隐藏状态
执行: python scripts/train_safety_prober.py

Safety Prober 是整个防御系统的第一道防线，用于检测输入的风险程度
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

# 导入模型定义
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.safety_prober import SafetyProber


class HiddenStateDataset(Dataset):
    """隐藏状态数据集"""
    
    def __init__(self, split_path: Path, layer_idx: int = -1):
        """
        Args:
            split_path: 数据划分JSON文件路径
            layer_idx: 使用哪一层的隐藏状态
        """
        with open(split_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.samples = []
        self.layer_idx = layer_idx
        
        # 加载越狱样本 (label=1, 有害)
        for item in self.data.get("jailbreak", []):
            self.samples.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "label": 1
            })
        
        # 加载良性样本 (label=0, 无害)
        for item in self.data.get("benign", []):
            self.samples.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "label": 0
            })
        
        # 加载拒绝样本 (label=1, 视为需要拒绝的请求)
        for item in self.data.get("refusal", []):
            self.samples.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "label": 1
            })
        
        # 加载遵从样本 (label=0, 无害)
        for item in self.data.get("compliance", []):
            self.samples.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "label": 0
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length=512):
    """批处理函数"""
    prompts = [item["prompt"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    return inputs, labels


def train_epoch(
    model,
    llm_model,
    dataloader,
    optimizer,
    device,
    layer_idx,
    epoch
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for inputs, labels in pbar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            
            # 取最后一个token的隐藏状态
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(last_hidden)
        loss = F.cross_entropy(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total:.4f}"
        })
    
    return total_loss / len(dataloader), correct / total


def evaluate(
    model,
    llm_model,
    dataloader,
    device,
    layer_idx
):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # 获取隐藏状态
            outputs = llm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
            
            # 前向传播
            logits = model(last_hidden)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # 计算指标
    accuracy = correct / total
    
    # 计算FPR (False Positive Rate)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 计算TPR (True Positive Rate / Recall)
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr
    }


def main():
    parser = argparse.ArgumentParser(description="Train Safety Prober")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to use (-1 for last)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" [TRAIN] Safety Prober Training Script")
    print("=" * 60)
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    
    # 加载LLM
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
    
    # 获取隐藏维度
    hidden_dim = llm_model.config.hidden_size
    num_layers = llm_model.config.num_hidden_layers
    
    layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer + 1
    print(f"Hidden dim: {hidden_dim}")
    print(f"Using layer: {layer_idx}")
    
    # 加载数据
    print("\nLoading data...")
    train_dataset = HiddenStateDataset(Path("data/splits/train.json"))
    val_dataset = HiddenStateDataset(Path("data/splits/val.json"))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    # 创建Safety Prober
    print("\nCreating Safety Prober...")
    prober = SafetyProber(hidden_dim=hidden_dim, num_classes=2)
    prober = prober.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(prober.parameters(), lr=args.lr)
    
    # 训练
    print("\nStarting training...")
    best_acc = 0
    output_dir = Path("checkpoints/safety_prober")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            prober, llm_model, train_loader, optimizer, device, layer_idx, epoch
        )
        
        val_metrics = evaluate(prober, llm_model, val_loader, device, layer_idx)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val FPR: {val_metrics['fpr']:.4f}, Val TPR: {val_metrics['tpr']:.4f}")
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                "model_state_dict": prober.state_dict(),
                "epoch": epoch,
                "accuracy": best_acc,
                "layer_idx": layer_idx,
                "hidden_dim": hidden_dim,
                "model_name": args.model_name
            }, output_dir / "best_model.pt")
            print(f"  [SAVED] Best model (acc={best_acc:.4f})")
    
    # 最终评估
    print("\n" + "=" * 60)
    print(" Final Evaluation")
    print("=" * 60)
    
    # 加载最佳模型
    checkpoint = torch.load(output_dir / "best_model.pt")
    prober.load_state_dict(checkpoint["model_state_dict"])
    
    test_dataset = HiddenStateDataset(Path("data/splits/test.json"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    test_metrics = evaluate(prober, llm_model, test_loader, device, layer_idx)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test FPR: {test_metrics['fpr']:.4f}")
    print(f"Test TPR: {test_metrics['tpr']:.4f}")
    
    print("\n[OK] Training complete!")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
