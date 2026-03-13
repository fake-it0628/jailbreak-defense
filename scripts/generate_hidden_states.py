# -*- coding: utf-8 -*-
"""
隐藏状态提取脚本
目的: 从基础模型中提取输入对应的隐藏状态
执行: python scripts/generate_hidden_states.py --model_name Qwen/Qwen2.5-0.5B-Instruct

这是整个项目的关键步骤，因为隐藏状态是后续所有分析的基础
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="提取LLM隐藏状态")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face模型名称"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="批处理大小"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="最大token长度"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="设备 (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="要提取的层 (all/10,15,20/last)"
    )
    return parser.parse_args()


def load_model(model_name: str, device: str):
    """加载模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n加载模型: {model_name}")
    
    # 确定设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,  # 关键！
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    print(f"模型层数: {num_layers}")
    
    return model, tokenizer, device, num_layers


def extract_hidden_states(
    model, 
    tokenizer, 
    texts: List[str], 
    device: str,
    layers: List[int],
    max_length: int = 512
) -> Dict:
    """
    提取文本的隐藏状态
    
    返回:
        Dict: {
            "layer_X": tensor of shape (batch, hidden_dim),  # 最后token的隐藏状态
            ...
        }
    """
    # 编码
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"]
    
    # 获取每个样本最后一个非padding token的位置
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    
    result = {}
    for layer_idx in layers:
        layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
        
        # 取最后一个token的隐藏状态
        batch_indices = torch.arange(layer_hidden.size(0))
        last_token_hidden = layer_hidden[batch_indices, seq_lengths]  # (batch, hidden_dim)
        
        result[f"layer_{layer_idx}"] = last_token_hidden.cpu()
    
    return result


def process_dataset(
    model,
    tokenizer,
    data: List[Dict],
    device: str,
    layers: List[int],
    batch_size: int,
    max_length: int,
    desc: str = "Processing"
):
    """处理整个数据集"""
    all_hidden_states = {f"layer_{l}": [] for l in layers}
    all_ids = []
    
    for i in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[i:i+batch_size]
        texts = [item.get("prompt", "") for item in batch]
        ids = [item.get("id", f"item_{i+j}") for j, item in enumerate(batch)]
        
        hidden = extract_hidden_states(
            model, tokenizer, texts, device, layers, max_length
        )
        
        for key in all_hidden_states:
            all_hidden_states[key].append(hidden[key])
        all_ids.extend(ids)
    
    # 合并
    for key in all_hidden_states:
        all_hidden_states[key] = torch.cat(all_hidden_states[key], dim=0)
    
    return all_hidden_states, all_ids


def main():
    args = parse_args()
    
    print("=" * 60)
    print(" 🧠 隐藏状态提取脚本")
    print("=" * 60)
    
    # 加载模型
    model, tokenizer, device, num_layers = load_model(args.model_name, args.device)
    
    # 确定要提取的层
    if args.layers == "all":
        layers = list(range(num_layers + 1))  # 包括embedding层
    elif args.layers == "last":
        layers = [num_layers]
    else:
        layers = [int(l) for l in args.layers.split(",")]
    
    print(f"提取层: {layers}")
    
    # 输出目录
    model_short_name = args.model_name.split("/")[-1]
    output_dir = Path("data/hidden_states") / model_short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    splits_dir = Path("data/splits")
    
    for split in ["train", "val", "test"]:
        split_path = splits_dir / f"{split}.json"
        if not split_path.exists():
            print(f"⚠ 跳过 {split}: 文件不存在")
            continue
        
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        for category in ["jailbreak", "benign", "refusal", "compliance"]:
            if category not in split_data:
                continue
            
            data = split_data[category]
            if not data:
                continue
            
            print(f"\n处理 {split}/{category}: {len(data)} 样本")
            
            hidden_states, ids = process_dataset(
                model, tokenizer, data, device, layers,
                args.batch_size, args.max_length,
                desc=f"{split}/{category}"
            )
            
            # 保存
            save_data = {
                "ids": ids,
                "hidden_states": {k: v.numpy().tolist() for k, v in hidden_states.items()},
                "metadata": {
                    "model": args.model_name,
                    "category": category,
                    "split": split,
                    "num_samples": len(ids),
                    "layers": layers,
                    "hidden_dim": hidden_states[f"layer_{layers[0]}"].shape[-1]
                }
            }
            
            save_path = split_output_dir / f"{category}_hidden_states.json"
            with open(save_path, 'w') as f:
                json.dump(save_data, f)
            
            print(f"✓ 已保存: {save_path}")
            
            # 也保存为pt文件（更高效）
            pt_path = split_output_dir / f"{category}_hidden_states.pt"
            torch.save({
                "ids": ids,
                "hidden_states": hidden_states,
                "metadata": save_data["metadata"]
            }, pt_path)
            print(f"✓ 已保存: {pt_path}")
    
    print("\n" + "=" * 60)
    print(" ✓ 隐藏状态提取完成!")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print(f"\n下一步: python scripts/train_safety_prober.py")


if __name__ == "__main__":
    main()
