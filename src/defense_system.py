# -*- coding: utf-8 -*-
"""
Jailbreak Defense System
完整的越狱防御系统，集成所有防御模块

核心流程:
1. Safety Prober 检测输入风险
2. Risk Encoder 累积多轮风险
3. Steering Matrix 对高风险输入进行干预
4. 生成安全的响应
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .models.safety_prober import SafetyProber
from .models.steering_matrix import SteeringMatrix, AdaptiveSteeringController
from .models.risk_encoder import RiskEncoder, MultiTurnRiskMemory


@dataclass
class DefenseResult:
    """防御系统结果"""
    is_harmful: bool
    risk_score: float
    accumulated_risk: float
    action_taken: str  # "none", "steered", "blocked"
    original_response: Optional[str] = None
    safe_response: Optional[str] = None
    debug_info: Optional[Dict] = None


class JailbreakDefenseSystem(nn.Module):
    """
    越狱防御系统
    
    集成所有防御模块，提供完整的防御能力
    """
    
    def __init__(
        self,
        hidden_dim: int,
        safety_prober: Optional[SafetyProber] = None,
        steering_matrix: Optional[SteeringMatrix] = None,
        risk_encoder: Optional[RiskEncoder] = None,
        risk_threshold: float = 0.5,
        block_threshold: float = 0.85,
        enable_steering: bool = True,
        enable_multi_turn: bool = True
    ):
        """
        Args:
            hidden_dim: LLM隐藏层维度
            safety_prober: Safety Prober模块
            steering_matrix: Steering Matrix模块
            risk_encoder: Risk Encoder模块
            risk_threshold: 风险阈值 (超过则干预)
            block_threshold: 阻断阈值 (超过则直接阻断)
            enable_steering: 是否启用激活空间干预
            enable_multi_turn: 是否启用多轮风险记忆
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.risk_threshold = risk_threshold
        self.block_threshold = block_threshold
        self.enable_steering = enable_steering
        self.enable_multi_turn = enable_multi_turn
        
        # 初始化模块
        self.safety_prober = safety_prober or SafetyProber(hidden_dim)
        self.steering_matrix = steering_matrix or SteeringMatrix(hidden_dim)
        self.risk_encoder = risk_encoder or RiskEncoder(hidden_dim)
        
        # 自适应强度控制器
        self.strength_controller = AdaptiveSteeringController(
            min_strength=0.0,
            max_strength=2.0,
            risk_threshold=risk_threshold
        )
        
        # 多轮风险记忆
        self.risk_memory = MultiTurnRiskMemory(
            decay_rate=0.9,
            accumulation_rate=0.3,
            risk_threshold=risk_threshold
        )
        
        # 标准拒绝响应模板
        self.refusal_templates = [
            "I cannot help with this request as it may be harmful.",
            "I'm not able to assist with that. Let me help with something else.",
            "This request goes against my guidelines. How else can I help you?",
        ]
    
    def reset_conversation(self):
        """重置对话状态"""
        self.risk_memory.reset()
    
    def analyze_risk(
        self,
        hidden_states: torch.Tensor,
        conversation_history: Optional[List[torch.Tensor]] = None
    ) -> Tuple[float, float]:
        """
        分析输入的风险程度
        
        Args:
            hidden_states: 当前输入的隐藏状态 (batch=1, hidden_dim)
            conversation_history: 对话历史的隐藏状态列表
        
        Returns:
            current_risk: 当前输入的风险分数
            accumulated_risk: 累积风险分数
        """
        # Safety Prober 检测当前风险
        with torch.no_grad():
            current_risk = self.safety_prober.get_risk_score(hidden_states).item()
        
        # 多轮风险累积
        if self.enable_multi_turn and conversation_history:
            # 构建序列
            seq = torch.stack(conversation_history + [hidden_states.squeeze(0)], dim=0)
            seq = seq.unsqueeze(0)  # (1, seq_len, hidden_dim)
            
            with torch.no_grad():
                multi_turn_risk = self.risk_encoder.get_risk_score(seq).item()
            
            # 综合当前风险和多轮风险
            combined_risk = max(current_risk, multi_turn_risk)
            accumulated_risk, _ = self.risk_memory.update(combined_risk)
        else:
            accumulated_risk, _ = self.risk_memory.update(current_risk)
        
        return current_risk, accumulated_risk
    
    def intervene(
        self,
        hidden_states: torch.Tensor,
        risk_score: float
    ) -> Tuple[torch.Tensor, float]:
        """
        对隐藏状态进行干预
        
        Args:
            hidden_states: 原始隐藏状态
            risk_score: 风险分数
        
        Returns:
            steered_states: 干预后的隐藏状态
            steering_strength: 使用的干预强度
        """
        if not self.enable_steering:
            return hidden_states, 0.0
        
        # 计算自适应干预强度
        strength = self.strength_controller.compute_strength(risk_score)
        
        if strength <= 0:
            return hidden_states, 0.0
        
        # 应用干预
        steered_states = self.steering_matrix.steer_with_direction(
            hidden_states, steering_strength=strength
        )
        
        return steered_states, strength
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        conversation_history: Optional[List[torch.Tensor]] = None,
        return_debug: bool = False
    ) -> DefenseResult:
        """
        完整的防御流程
        
        Args:
            hidden_states: 当前输入的隐藏状态
            conversation_history: 对话历史
            return_debug: 是否返回调试信息
        
        Returns:
            DefenseResult: 防御结果
        """
        debug_info = {} if return_debug else None
        
        # 1. 风险分析
        current_risk, accumulated_risk = self.analyze_risk(
            hidden_states, conversation_history
        )
        
        if return_debug:
            debug_info["current_risk"] = current_risk
            debug_info["accumulated_risk"] = accumulated_risk
            debug_info["risk_trend"] = self.risk_memory.get_trend()
        
        # 2. 决策
        max_risk = max(current_risk, accumulated_risk)
        
        # 高风险: 直接阻断
        if max_risk >= self.block_threshold:
            return DefenseResult(
                is_harmful=True,
                risk_score=current_risk,
                accumulated_risk=accumulated_risk,
                action_taken="blocked",
                safe_response=self.refusal_templates[0],
                debug_info=debug_info
            )
        
        # 中等风险: 干预
        if max_risk >= self.risk_threshold:
            steered_states, strength = self.intervene(hidden_states, max_risk)
            
            if return_debug:
                debug_info["steering_strength"] = strength
            
            return DefenseResult(
                is_harmful=True,
                risk_score=current_risk,
                accumulated_risk=accumulated_risk,
                action_taken="steered",
                debug_info=debug_info
            )
        
        # 低风险: 放行
        return DefenseResult(
            is_harmful=False,
            risk_score=current_risk,
            accumulated_risk=accumulated_risk,
            action_taken="none",
            debug_info=debug_info
        )
    
    def save(self, path: str):
        """保存防御系统"""
        torch.save({
            "safety_prober": self.safety_prober.state_dict(),
            "steering_matrix": self.steering_matrix.state_dict(),
            "risk_encoder": self.risk_encoder.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "risk_threshold": self.risk_threshold,
                "block_threshold": self.block_threshold,
                "enable_steering": self.enable_steering,
                "enable_multi_turn": self.enable_multi_turn
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "JailbreakDefenseSystem":
        """加载防御系统"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        system = cls(
            hidden_dim=config["hidden_dim"],
            risk_threshold=config["risk_threshold"],
            block_threshold=config["block_threshold"],
            enable_steering=config["enable_steering"],
            enable_multi_turn=config["enable_multi_turn"]
        )
        
        system.safety_prober.load_state_dict(checkpoint["safety_prober"])
        system.steering_matrix.load_state_dict(checkpoint["steering_matrix"])
        system.risk_encoder.load_state_dict(checkpoint["risk_encoder"])
        
        return system.to(device)


def create_defense_system(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    checkpoint_dir: str = "checkpoints",
    device: str = "auto"
) -> Tuple[JailbreakDefenseSystem, any, any]:
    """
    创建防御系统的便捷函数
    
    Args:
        model_name: LLM模型名称
        checkpoint_dir: 检查点目录
        device: 设备
    
    Returns:
        defense_system: 防御系统
        llm_model: LLM模型
        tokenizer: 分词器
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pathlib import Path
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,
        trust_remote_code=True
    )
    
    if device == "cpu":
        llm_model = llm_model.to(device)
    
    llm_model.eval()
    
    hidden_dim = llm_model.config.hidden_size
    
    # 创建防御系统
    defense_system = JailbreakDefenseSystem(hidden_dim=hidden_dim)
    
    # 尝试加载预训练权重
    checkpoint_path = Path(checkpoint_dir)
    
    prober_path = checkpoint_path / "safety_prober" / "best_model.pt"
    if prober_path.exists():
        checkpoint = torch.load(prober_path, map_location=device)
        defense_system.safety_prober.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Loaded Safety Prober from {prober_path}")
    
    defense_system = defense_system.to(device)
    
    return defense_system, llm_model, tokenizer
